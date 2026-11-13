"""无权重下载的协议/掩码/指标测试；不把这些测试冒充远程 GPU 训练验收。"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from llm_lab.evaluation.task_eval import aggregate_scores, check_eval_isolation, parse_task_json, score_one, score_predictions
from llm_lab.workflows.posttrain import (
    DEFAULTS, encode_example, format_prompt, make_manifest, make_training_plan,
    read_task_rows, resolve_resume, validate_config, write_json,
)


GOLD = '{"intent":"alarm_set","slots":[{"slot":"time","value":"five am"}]}'


def row(source_id="train-1", language="en", prompt="set an alarm", **extra):
    return {"source_id": source_id, "language": language, "prompt": prompt, "completion": GOLD, **extra}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")


@pytest.fixture
def config(tmp_path, monkeypatch):
    # 其它并行开发任务可能正在改源码；本协议夹具固定代码身份，专门用例再模拟变更。
    monkeypatch.setattr("llm_lab.workflows.posttrain_contracts.source_fingerprint", lambda: "fixture-source-code")
    write_rows(tmp_path / "data/train.jsonl", [row()])
    write_rows(tmp_path / "data/validation.jsonl", [row("val-1", "zh", "查询天气")])
    path = tmp_path / "config.json"
    payload = {"project_root": str(tmp_path), "stage": "sft", "method": "lora", "train_file": "data/train.jsonl",
               "validation_file": "data/validation.jsonl", "output_dir": "runs/sft", "max_length": 1024}
    write_json(path, payload)
    return path, payload


def test_validate_only_needs_no_hf_or_torch_and_creates_no_output(config):
    path, _ = config
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    # -S 不加载 site-packages；子进程还断言所有训练依赖没有被导入。
    code = "from llm_lab.workflows.posttrain import main; import sys; main(['--config',sys.argv[1],'--validate-only']); assert not any(n in sys.modules for n in ['torch','transformers','peft','trl','datasets'])"
    result = subprocess.run([sys.executable, "-S", "-c", code, str(path)], env=env, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert '"valid": true' in result.stdout
    assert not (path.parent / "runs/sft").exists()


@pytest.mark.parametrize("field,value,match", [
    ("unknown_field", 1, "未知配置"), ("base_revision", "main", "40 位"),
    ("base_revision", "060db64", "40 位"), ("batch_size", True, "正整数"),
    ("precision", "bf8", "precision"), ("budget_ledger", "another/ledger.json", "共享"),
    ("base_model", "Qwen/Qwen2.5-0.5B-Instruct", "true Base"),
])
def test_rejects_bad_config(config, field, value, match):
    path, payload = config
    write_json(path, {**payload, field: value})
    with pytest.raises(ValueError, match=match):
        validate_config(path)


def test_rejects_missing_path_and_test_as_train(config):
    path, payload = config
    write_json(path, {**payload, "train_file": "data/missing.jsonl"})
    with pytest.raises(ValueError, match="存在"):
        validate_config(path)
    write_rows(path.parent / "data/test.jsonl", [row()])
    write_json(path, {**payload, "train_file": "data/test.jsonl"})
    with pytest.raises(ValueError, match="test"):
        validate_config(path)


def test_schema_and_cross_language_source_leakage(config):
    path, _ = config
    write_rows(path.parent / "data/validation.jsonl", [row("train-1", "zh", "完全不同的中文")])
    with pytest.raises(ValueError, match="泄漏"):
        validate_config(path)
    write_rows(path.parent / "data/validation.jsonl", [row("val-1", "en", " SET   AN ALARM ")])
    with pytest.raises(ValueError, match="泄漏"):
        validate_config(path)
    write_rows(path.parent / "data/validation.jsonl", [{**row("val-1"), "messages": []}])
    with pytest.raises(ValueError, match="字段必须"):
        validate_config(path)


class TinyTokenizer:
    eos_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) + 1 for char in text]


def test_completion_only_mask_and_eos_without_chat_template():
    sample = row()
    encoded = encode_example(sample, TinyTokenizer(), 1024, "sft")
    boundary = len(format_prompt(sample["prompt"]))
    assert encoded["labels"][:boundary] == [-100] * boundary
    assert encoded["labels"][boundary:] == TinyTokenizer.encode(GOLD) + [0]
    assert encoded["input_ids"][-1] == encoded["labels"][-1] == 0
    assert all(value == 1 for value in encoded["attention_mask"])
    with pytest.raises(ValueError, match="静默截断"):
        encode_example(sample, TinyTokenizer(), 10, "sft")


def test_token_plan_is_deterministic_and_stays_under_budget():
    encoded = [encode_example(row(str(i), prompt="question " + "x" * i), TinyTokenizer(), 1024, "sft") for i in range(4)]
    first = make_training_plan(encoded, token_budget=2000, effective_batch_size=4, seed=42, stage="sft")
    second = make_training_plan(encoded, token_budget=2000, effective_batch_size=4, seed=42, stage="sft")
    assert first == second
    indices, plan = first
    assert len(indices) % 4 == 0
    assert plan["input_tokens"] == sum(len(encoded[i]["input_ids"]) for i in indices) <= 2000
    assert plan["loss_tokens"] == len(indices) * (len(GOLD) + 1)
    with pytest.raises(ValueError, match="不足"):
        make_training_plan(encoded, token_budget=1, effective_batch_size=4, seed=42, stage="sft")


def test_language_balance_uses_supervised_tokens_not_example_counts():
    short = {"input_ids": [1] * 3, "labels": [-100, 1, 2]}
    long = {"input_ids": [1] * 41, "labels": [-100] + [2] * 40}
    kwargs = dict(token_budget=10000, effective_batch_size=4, seed=3, stage="sft", languages=["zh", "en"])
    indices, plan = make_training_plan([short, long], **kwargs)
    assert (indices, plan) == make_training_plan([short, long], **kwargs)
    assert indices.count(0) > 10 * indices.count(1)
    language_loss = plan["loss_tokens_by_language"]
    assert abs(language_loss["zh"] - language_loss["en"]) <= plan["max_language_loss_token_difference"] == 40
    assert plan["loss_token_relative_imbalance"] < 0.01
    assert sum(plan["input_tokens_by_language"].values()) == plan["input_tokens"]
    with pytest.raises(ValueError, match="同时包含"):
        make_training_plan([short], **{**kwargs, "languages": ["zh"]})


def make_artifact(path, manifest, *, checkpoint=False):
    path.mkdir(parents=True, exist_ok=True)
    if manifest["checkpoint_type"] == "adapter":
        (path / "adapter_model.safetensors").write_bytes(b"metadata-only-fixture")
        write_json(path / "adapter_config.json", {"base_model_name_or_path": manifest["base_model"]})
    else:
        (path / "model.safetensors").write_bytes(b"metadata-only-fixture")
        write_json(path / "config.json", {})
    write_json(path / "run_manifest.json", {**manifest, "complete_checkpoint": checkpoint})


def test_resume_checks_real_identity_data_hash_and_full_state(config):
    path, _ = config
    run = validate_config(path)
    checkpoint = Path(run.config["output_dir"]) / "checkpoint-1"
    manifest = {**make_manifest(run), "training_plan": {"plan_hash": "test-plan"}}
    make_artifact(checkpoint, manifest, checkpoint=True)
    write_json(checkpoint / "trainer_state.json", {"global_step": 1})
    write_json(checkpoint / "progress.json", {"global_step": 1, "examples_consumed": 16, "plan_hash": "test-plan"})
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"state-fixture")
    assert validate_config(path, str(checkpoint)).resume_path == checkpoint
    assert resolve_resume(Path(run.config["output_dir"]), "latest") == checkpoint
    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(ValueError, match="完整恢复"):
        validate_config(path, str(checkpoint))
    (checkpoint / "optimizer.pt").write_bytes(b"state-fixture")
    write_rows(path.parent / "data/train.jsonl", [row(prompt="different training input")])
    with pytest.raises(ValueError, match="身份变化"):
        validate_config(path, str(checkpoint))


def test_resume_rejects_export_only_and_corrupt_data_position(config):
    path, _ = config
    run = validate_config(path)
    export = path.parent / "export"
    make_artifact(export, make_manifest(run))
    with pytest.raises(ValueError, match="完整训练 checkpoint"):
        validate_config(path, str(export))


def test_resume_rejects_changed_source_fingerprint(config, monkeypatch):
    path, _ = config
    run = validate_config(path)
    checkpoint = Path(run.config["output_dir"]) / "checkpoint-1"
    make_artifact(checkpoint, make_manifest(run), checkpoint=True)
    monkeypatch.setattr("llm_lab.workflows.posttrain_contracts.source_fingerprint", lambda: "different-source-code")
    with pytest.raises(ValueError, match="身份变化"):
        validate_config(path, str(checkpoint))


def test_dpo_requires_sft_and_preserves_adapter_reference_metadata(config):
    path, payload = config
    train = {k: v for k, v in row().items() if k != "completion"}
    val = {k: v for k, v in row("val-1", "zh", "天气查询").items() if k != "completion"}
    sft_run = validate_config(path)
    source = path.parent / "sft_export"
    make_artifact(source, make_manifest(sft_run))
    write_rows(path.parent / "data/dpo_train.jsonl", [{**train, "chosen": GOLD, "rejected": "broken JSON"}])
    write_rows(path.parent / "data/dpo_validation.jsonl", [{**val, "chosen": GOLD, "rejected": '{"intent":"wrong","slots":[]}'}])
    dpo = {**payload, "stage": "dpo", "train_file": "data/dpo_train.jsonl", "validation_file": "data/dpo_validation.jsonl"}
    write_json(path, dpo)
    with pytest.raises(ValueError, match="sft_checkpoint"):
        validate_config(path)
    write_json(path, {**dpo, "sft_checkpoint": str(source)})
    run = validate_config(path)
    manifest = make_manifest(run)
    assert run.source_manifest["checkpoint_type"] == "adapter"
    assert manifest["reference_semantics"] == "frozen_sft_start"
    assert manifest["sft_checkpoint"] == str(source)
    assert manifest["weights_base_checkpoint"] is None


def test_slot_multiset_invalid_schema_and_language_macro():
    duplicate = '{"intent":"alarm_set","slots":[{"slot":"time","value":"five am"},{"slot":"time","value":"five am"}]}'
    score = score_one(GOLD, duplicate)
    assert (score["slot_tp"], score["slot_fp"], score["slot_fn"]) == (1, 1, 0)
    assert score["intent_correct"] and not score["task_success"]
    malformed_schema = score_one(GOLD, '{"intent":"alarm_set","slots":[],"explanation":"extra"}')
    assert malformed_schema["json_parse_valid"] and not malformed_schema["json_valid"]
    with pytest.raises(ValueError, match="重复字段"):
        parse_task_json('{"intent":"a","intent":"b","slots":[]}')
    records = [{"language": "zh", **score_one(GOLD, GOLD)}] + [{"language": "en", **score_one(GOLD, "broken")}] * 3
    metrics = aggregate_scores(records)
    assert metrics["pooled"]["intent_accuracy"] == 0.25
    assert metrics["macro_languages"]["intent_accuracy"] == 0.5
    assert metrics["zh"]["task_success_rate"] == 1
    assert metrics["en"]["slot_micro_f1"] == 0


def test_predictions_align_exact_ids_and_test_isolation():
    rows = [row()]
    with pytest.raises(ValueError, match="ID 集合"):
        score_predictions(rows, [{"source_id": "wrong", "language": "en", "prediction": GOLD}])
    with pytest.raises(ValueError, match="重叠"):
        check_eval_isolation(rows, {"validation_source_ids": ["train-1"]}, "test")
    check_eval_isolation(rows, {"validation_source_ids": ["train-1"]}, "validation")


def test_test_split_requires_explicit_final_and_offline_scoring(config):
    from llm_lab.evaluation.task_eval import main
    path, _ = config
    predictions = path.parent / "predictions.jsonl"
    write_rows(predictions, [{"source_id": "train-1", "language": "en", "prediction": GOLD}])
    output = path.parent / "scoring"
    arguments = ["--predictions", str(predictions), "--data", str(path.parent / "data/train.jsonl"),
                 "--split", "test", "--output", str(output)]
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2
    main(arguments + ["--final"])
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["metrics"]["en"]["intent_accuracy"] == 1.0
    assert metrics["metrics"]["macro_languages"]["intent_accuracy"] is None
    assert metrics["isolation_checked"] is False  # 离线文件没有模型训练来源，不冒充已查过。


def test_gpu_eval_refuses_unknown_price_before_loading_any_model(tmp_path):
    from llm_lab.evaluation.task_eval import generate_predictions
    with pytest.raises(ValueError, match="实际单价"):
        generate_predictions(tmp_path, [], {"config": {}}, tmp_path, 32, None, 1.0)


def _install_adapter_restore_stub(monkeypatch, tmp_path, *, replace_parameter=False):
    """没有 PEFT 时也验证恢复分支；不把 stub 当成真实 HF 兼容性证据。"""
    from types import ModuleType, SimpleNamespace
    parameters = {
        "layer.lora_B.default.weight": SimpleNamespace(value=-1.0, requires_grad=True),
        "layer.lora_B.reference.weight": SimpleNamespace(value=999.0, requires_grad=False),
    }

    class Model:
        peft_config = {"default": {}, "reference": {}}
        active_adapter = "default"

        def named_parameters(self):
            return parameters.items()

        def set_adapter(self, name):
            self.active_adapter = name
            for key, parameter in parameters.items():
                parameter.requires_grad = f".{name}." in key

    checkpoint, sft = tmp_path / "checkpoint", tmp_path / "sft"
    for path in (checkpoint, sft):
        path.mkdir()
        (path / "adapter_model.safetensors").write_bytes(b"stub-only")
    calls = []

    def load_weights(path, *, device, local_files_only):
        assert device == "cpu" and local_files_only is True
        calls.append(path)
        return {"value": 7.0 if Path(path) == checkpoint else 3.0}

    def set_weights(model, state, *, adapter_name, low_cpu_mem_usage):
        assert low_cpu_mem_usage is False
        key = f"layer.lora_B.{adapter_name}.weight"
        if replace_parameter:
            parameters[key] = SimpleNamespace(value=state["value"], requires_grad=adapter_name == "default")
        else:
            parameters[key].value = state["value"]
        return SimpleNamespace(unexpected_keys=[], missing_keys=[])

    package, utils, module = ModuleType("peft"), ModuleType("peft.utils"), ModuleType("peft.utils.save_and_load")
    package.__path__, utils.__path__ = [], []
    module.load_peft_weights, module.set_peft_model_state_dict = load_weights, set_weights
    for key, value in (("peft", package), ("peft.utils", utils), ("peft.utils.save_and_load", module)):
        monkeypatch.setitem(sys.modules, key, value)
    return Model(), parameters, checkpoint, sft, calls


def test_dpo_adapter_restore_stub_uses_checkpoint_policy_and_original_sft_reference(monkeypatch, tmp_path):
    from llm_lab.workflows.posttrain_runtime import restore_dpo_adapters_inplace
    model, parameters, checkpoint, sft, calls = _install_adapter_restore_stub(monkeypatch, tmp_path)
    optimizer_parameter = parameters["layer.lora_B.default.weight"]
    identities = {key: id(value) for key, value in parameters.items()}
    restore_dpo_adapters_inplace(model, checkpoint, sft)
    assert optimizer_parameter is parameters["layer.lora_B.default.weight"]
    assert optimizer_parameter.value == 7.0
    assert parameters["layer.lora_B.reference.weight"].value == 3.0
    assert not parameters["layer.lora_B.reference.weight"].requires_grad
    assert model.active_adapter == "default"
    assert identities == {key: id(value) for key, value in parameters.items()}
    assert calls == [str(checkpoint), str(sft)]  # 不读取 checkpoint/reference 副本。


def test_dpo_adapter_restore_stub_without_named_reference(monkeypatch, tmp_path):
    from llm_lab.workflows.posttrain_runtime import restore_dpo_adapters_inplace
    model, parameters, checkpoint, _, calls = _install_adapter_restore_stub(monkeypatch, tmp_path)
    restore_dpo_adapters_inplace(model, checkpoint)
    assert parameters["layer.lora_B.default.weight"].value == 7.0
    assert parameters["layer.lora_B.reference.weight"].value == 999.0
    assert calls == [str(checkpoint)]


def test_dpo_adapter_restore_rejects_replacing_optimizer_parameters(monkeypatch, tmp_path):
    from llm_lab.workflows.posttrain_runtime import restore_dpo_adapters_inplace
    model, _, checkpoint, sft, _ = _install_adapter_restore_stub(monkeypatch, tmp_path, replace_parameter=True)
    with pytest.raises(RuntimeError, match="替换了 Parameter"):
        restore_dpo_adapters_inplace(model, checkpoint, sft)


def test_tiny_peft_dpo_checkpoint_restores_policy_and_frozen_reference_cpu(monkeypatch, tmp_path):
    """真实随机 Qwen2/PEFT CPU 回归；远程装齐锁定依赖后自动执行，无模型下载。"""
    from importlib import metadata
    from llm_lab.workflows.posttrain_contracts import PINNED_PACKAGES
    for package in ("transformers", "peft"):
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = None
        if actual != PINNED_PACKAGES[package]:
            reason = f"真实 CPU adapter 回归需要 {package}=={PINNED_PACKAGES[package]}，当前 {actual}"
            if os.environ.get("LLM_LAB_REQUIRE_PEFT_TEST") == "1":
                pytest.fail(reason)
            pytest.skip(reason)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    def forbidden_network(*args, **kwargs):
        raise AssertionError("tiny CPU 回归禁止联网")

    monkeypatch.setattr("requests.sessions.Session.request", forbidden_network)
    monkeypatch.setattr("urllib.request.urlopen", forbidden_network)
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict
    from llm_lab.workflows.posttrain_runtime import restore_dpo_adapters_inplace

    def base_model():
        torch.manual_seed(17)
        return Qwen2ForCausalLM(Qwen2Config(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32, bos_token_id=1, eos_token_id=1, pad_token_id=0))

    model = get_peft_model(base_model(), LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B.default" in name:
                parameter.fill_(0.125)  # SFT adapter 必须非零，不能把原始 Base 混作 reference。
    source, checkpoint = tmp_path / "sft", tmp_path / "checkpoint-1"
    model.save_pretrained(source, selected_adapters=["default"], save_embedding_layers=False)
    expected_sft = {key: value.clone() for key, value in get_peft_model_state_dict(model, save_embedding_layers=False).items()}
    model.load_adapter(source, adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".default." in name:
                parameter.add_(0.25)  # 模拟 DPO 已经改变 policy。
    expected_policy = {key: value.clone() for key, value in get_peft_model_state_dict(model, save_embedding_layers=False).items()}
    model.save_pretrained(checkpoint, save_embedding_layers=False)
    assert (checkpoint / "adapter_model.safetensors").is_file()
    assert (checkpoint / "reference/adapter_model.safetensors").is_file()

    fresh = PeftModel.from_pretrained(base_model(), source, is_trainable=True)
    fresh.load_adapter(source, adapter_name="reference", is_trainable=False)
    fresh.set_adapter("default")
    parameters_before = {name: id(parameter) for name, parameter in fresh.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad], lr=1e-3)
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    trainer = Trainer(model=fresh, args=TrainingArguments(output_dir=str(tmp_path / "runner"), use_cpu=True,
        report_to=[], bf16=False, fp16=False, save_strategy="no", dataloader_pin_memory=False))
    trainer._load_from_checkpoint(str(checkpoint))
    # 真实 v4.57.1 的子目录分支漏载 default：这里应仍为原 SFT，而非训练后的 policy。
    before_fix = get_peft_model_state_dict(fresh, save_embedding_layers=False)
    assert any(not torch.equal(before_fix[key], expected_policy[key]) for key in expected_policy)
    restore_dpo_adapters_inplace(fresh, checkpoint, source)
    restored_policy = get_peft_model_state_dict(fresh, save_embedding_layers=False)
    restored_reference = get_peft_model_state_dict(fresh, adapter_name="reference", save_embedding_layers=False)
    assert all(torch.equal(restored_policy[key], value) for key, value in expected_policy.items())
    assert all(torch.equal(restored_reference[key], value) for key, value in expected_sft.items())
    assert parameters_before == {name: id(parameter) for name, parameter in fresh.named_parameters()}
    assert optimizer_ids == [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert all(not parameter.requires_grad for name, parameter in fresh.named_parameters() if ".reference." in name)

    optimizer.zero_grad()
    input_ids = torch.tensor([[2, 3, 4, 1]])
    loss = fresh(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    after_update = get_peft_model_state_dict(fresh, adapter_name="reference", save_embedding_layers=False)
    assert all(torch.equal(after_update[key], value) for key, value in expected_sft.items())
