"""真实 tiny Decoder 的 SFT 更新、恢复和身份边界测试；只使用 CPU fixture。"""
from copy import deepcopy
from dataclasses import asdict
import errno
import importlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch

from llm_lab.config import ModelConfig
from llm_lab.models import Decoder
from llm_lab.common.repro import rng_state
from llm_sft.config import read_config
from llm_sft.data import SFTCorpus, sha256_file
from test_sft_runtime_data import corpus, repin, save  # noqa: F401


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def engine():
    # 根代理实现引擎期间仍可独立收集此测试文件；执行时才读取稳定 API。
    return importlib.import_module("llm_sft.engine")


def model_config(vocab_size, dropout=.2):
    return ModelConfig(vocab_size=vocab_size, dim=16, n_layers=1, n_heads=2,
                       n_kv_heads=1, ffn_dim=32, max_seq_len=1024,
                       dropout=dropout, activation_checkpointing=False)


def add_tail_row(directory, manifest, rows, tokenizer):
    row = deepcopy(rows["train"][0])
    row.update(id="train-tail", group_id="train-tail", prompt="请给出另一句话。", response="这是额外的回答。")
    prefix = tokenizer.encode("User:\n"+row["prompt"]+"\n\nAssistant:\n", add_special_tokens=False).ids
    answer = tokenizer.encode(row["response"], add_special_tokens=False).ids+[2]
    row.update(input_ids=prefix+answer, loss_mask=[0]*len(prefix)+[1]*len(answer),
               prompt_token_count=len(prefix), answer_token_count=len(answer))
    rows["train"].append(row)
    rows["pilot"] = [{**deepcopy(item), "split":"pilot"} for item in rows["train"]]
    repin(directory, manifest, rows)


def fixture_config(corpus, output_name="run", *, tail=False, dropout=.2):
    directory, manifest, rows, tokenizer = corpus
    if tail:
        add_tail_row(directory, manifest, rows, tokenizer)
    # Base 的来源 manifest 保持冻结，后续更改 SFT 数据不会假装更改 Base 来源。
    base_manifest = directory/"base_manifest.json"
    base_manifest.write_bytes((directory/"manifest.json").read_bytes())
    torch.manual_seed(7)
    spec = model_config(tokenizer.get_vocab_size(), dropout)
    model = Decoder(spec)
    base = directory/"base.pt"
    torch.save({"schema_version":1,"stage":"pretrain","config":{"model":asdict(spec),
                 "data_manifest":str(base_manifest),"project_root":str(directory)},
                "identity":{"tokenizer_sha256":manifest["tokenizer"]["sha256"],
                            "manifest_sha256":sha256_file(base_manifest)},
                "model":model.state_dict(),"step":10,"tokens_seen":1000},base)
    raw = {"schema_version":1,"stage":"sft","method":"full","project_root":str(directory),
           "base_checkpoint":"base.pt","data_manifest":"manifest.json","output_dir":output_name,
           "train_split":"train","fixture":True,"min_free_bytes":0,"experiment":{"purpose":"CPU test"},
           "base_checkpoint_sha256":sha256_file(base),"data_manifest_sha256":sha256_file(directory/"manifest.json"),
           "data_identity":manifest["data_identity"],"runtime_sha256":None,
           "train":{"device":"cpu","dtype":"fp32","micro_batch_size":1,"gradient_accumulation_steps":2,
                    "epochs":3.0,"learning_rate":.005,"warmup_ratio":.25,"eval_interval":1,"save_interval":1,
                    "log_interval":1,"tensorboard":False,"wandb":False,"activation_checkpointing":False,
                    "eval_micro_batch_size":2,"retention_eval_batches":0},
           "budget":{"hourly_rate_cny":0,"run_limit_cny":10,"project_limit_cny":500,"reserve_cny":100,
                     "max_runtime_seconds":120,"ledger_path":"ledger.json"}}
    path = directory/(output_name+".json")
    save(path,raw)
    return read_config(path), base


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def assert_state_equal(left, right):
    if isinstance(left,torch.Tensor):
        assert isinstance(right,torch.Tensor) and left.dtype==right.dtype and torch.equal(left,right)
    elif isinstance(left,np.ndarray):
        assert isinstance(right,np.ndarray) and np.array_equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:assert_state_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert type(left) is type(right) and len(left)==len(right)
        for a,b in zip(left,right):assert_state_equal(a,b)
    else:
        assert left==right


class BrokenConsole(io.StringIO):
    """在指定 JSON 事件后模拟永久关闭的日志管道，不接触 pytest 的真实文件描述符。"""

    def __init__(self, trigger, error):
        super().__init__()
        self.trigger = trigger
        self.error = error
        self.broken = False

    def write(self, value):
        if not self.broken:
            try:
                event = json.loads(value)
            except (TypeError, ValueError):
                event = None
            self.broken = event is not None and self.trigger(event)
        if self.broken:
            raise self.error
        return super().write(value)


def broken_pipe_error(kind):
    if kind == "broken_pipe":
        # 某些自定义 stdout 只抛异常类型，没有 errno。
        return BrokenPipeError("console reader closed")
    # 避免 OSError(errno.EPIPE, ...) 自动转为 BrokenPipeError，单独覆盖 errno 分支。
    error = OSError("console reader closed")
    error.errno = errno.EPIPE
    return error


@pytest.mark.parametrize("error_kind", ["broken_pipe", "epipe"])
def test_console_json_survives_closed_stdout_without_fileno(monkeypatch, error_kind):
    module = engine()
    console = BrokenConsole(lambda event: True, broken_pipe_error(error_kind))
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", console)
        module._console_json({"phase": "stop_requested", "signal": signal.SIGTERM})
        module._console_json({"step": 2, "train_loss": 1.0})
    assert console.broken


def test_console_json_preserves_unrelated_io_errors(monkeypatch):
    console = BrokenConsole(lambda event: True, OSError(errno.ENOSPC, "disk full"))
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", console)
        with pytest.raises(OSError) as failure:
            engine()._console_json({"step": 1})
    assert failure.value.errno == errno.ENOSPC


@pytest.mark.parametrize("failure_event", ["training_log", "signal_log"])
@pytest.mark.parametrize("error_kind", ["broken_pipe", "epipe"])
def test_closed_stdout_keeps_current_update_full_checkpoint_and_resume(
        corpus, monkeypatch, failure_event, error_kind):
    config, _ = fixture_config(corpus, "broken-console")
    # 第 2 步前没有周期保存；必须依赖正常停止路径写出完整断点。
    config["train"].update(eval_interval=100, save_interval=100, log_interval=1)
    module = engine()
    original_update = module.train_update
    updates = []
    previous_handler = signal.getsignal(signal.SIGTERM)

    def update_with_stop(*args, **kwargs):
        updates.append(len(updates) + 1)
        if len(updates) == 2:
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler) and handler is not previous_handler
            # 直接调用当前引擎的 handler，避免给整个 pytest 进程发送真实信号。
            handler(signal.SIGTERM, None)
        return original_update(*args, **kwargs)

    def trigger(event):
        if failure_event == "training_log":
            return event.get("step") == 1 and "train_loss" in event
        return event.get("phase") == "stop_requested"

    console = BrokenConsole(trigger, broken_pipe_error(error_kind))
    monkeypatch.setattr(module, "train_update", update_with_stop)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", console)
        stopped = module.run_sft(config)

    assert console.broken
    assert stopped["status"] == "signal_stop" and stopped["step"] == 2
    assert updates == [1, 2]
    assert signal.getsignal(signal.SIGTERM) is previous_handler
    output = Path(config["output_dir"])
    checkpoint = output / "last.pt"
    saved = load(checkpoint)
    assert saved["checkpoint_kind"] == "full" and saved["step"] == 2
    assert saved["progress"]["next_update"] == 2
    assert {"model", "optimizer", "scheduler", "scaler", "rng"} <= saved.keys()
    assert saved["optimizer"]["state"]
    assert saved["scheduler"]["last_epoch"] == 2
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["step"] for item in metrics] == [0, 1, 2]
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"] == "signal_stop"

    # 在完整恢复状态上只补第 3 次更新，不能悄悄从 Base 重新跑三步。
    resumed = module.run_sft(config, resume=checkpoint)
    assert resumed["status"] == "completed" and resumed["step"] == 3
    assert updates == [1, 2, 3]
    assert load(checkpoint)["progress"]["next_update"] == 3
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["step"] for item in metrics] == [0, 1, 2, 3]


@pytest.mark.skipif(os.name != "posix", reason="真实关闭管道行为仅验证远程 Linux；Windows 映射 EINVAL")
def test_closed_os_pipe_keeps_checkpoint_and_exits_cleanly(corpus):
    config, _ = fixture_config(corpus, "broken-os-pipe")
    config["train"].update(eval_interval=100, save_interval=100, log_interval=1)
    config_path = Path(config["project_root"]) / "child-config.json"
    save(config_path, config)
    error_path = config_path.with_suffix(".error.txt")
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root / "src"), CUDA_VISIBLE_DEVICES="",
                       PYTHONDONTWRITEBYTECODE="1")
    code = ("import sys, torch, traceback\n"
            "from pathlib import Path\n"
            "torch.set_num_threads(1)\n"
            "from llm_sft.config import read_config\n"
            "from llm_sft.engine import run_sft\n"
            "try:\n"
            "    run_sft(read_config(sys.argv[1]))\n"
            "except BaseException:\n"
            "    Path(sys.argv[2]).write_text(traceback.format_exc(), encoding='utf-8')\n"
            "    raise\n")
    process = subprocess.Popen([sys.executable, "-u", "-c", code, str(config_path), str(error_path)],
                               cwd=root, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    # 只关闭子进程 stdout/stderr 共用管道的读端，宿主 pytest 的 FD 保持不变。
    process.stdout.close()
    try:
        returncode = process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    assert returncode == 0, error_path.read_text(encoding="utf-8") if error_path.exists() else returncode
    output = Path(config["output_dir"])
    saved = load(output / "last.pt")
    assert saved["checkpoint_kind"] == "full" and saved["step"] == 3
    assert saved["progress"]["next_update"] == 3
    assert saved["optimizer"]["state"]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["step"] for item in metrics] == [0, 1, 2, 3]


def test_update_weights_micro_losses_by_supervised_tokens(corpus):
    config,_ = fixture_config(corpus,dropout=0)
    config["train"]["max_grad_norm"] = 1e6
    data = SFTCorpus(config["data_manifest"],"train")
    batches = [data.collate([index]) for index in (0,1)]
    assert batches[0]["supervised_tokens"] != batches[1]["supervised_tokens"]
    torch.manual_seed(23)
    accumulated = Decoder(model_config(data.vocab_size,dropout=0))
    reference = deepcopy(accumulated)
    optimizer = torch.optim.SGD(accumulated.parameters(),lr=.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(),lr=.05)
    scaler = torch.amp.GradScaler("cuda",enabled=False)
    combined = data.collate([0,1])
    reference_loss = reference(combined["x"],combined["y"]).loss
    reference_loss.backward();reference_optimizer.step()
    result = engine().train_update(accumulated,optimizer,scaler,data,[[0],[1]],config)
    assert result["updated"] and result["examples"]==2
    assert result["supervised_tokens"]==combined["supervised_tokens"]
    assert result["input_tokens"]==combined["input_tokens"]
    assert result["loss"]==pytest.approx(float(reference_loss.detach()),rel=2e-6)
    for actual,expected in zip(accumulated.parameters(),reference.parameters()):
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)


def test_validation_reports_each_language_and_restores_mode_rng(corpus):
    config,_ = fixture_config(corpus,dropout=.2)
    data = SFTCorpus(config["data_manifest"],"validation")
    model = Decoder(model_config(data.vocab_size,dropout=.2)).train()
    before = rng_state()
    result = engine().evaluate_sft_model(model,data,config)
    assert model.training
    assert_state_equal(rng_state(),before)
    model.eval()
    with torch.no_grad():
        expected={}
        for language in ("en","zh"):
            batch=data.collate(data.language_indices[language])
            expected[language]=float(model(batch["x"],batch["y"]).loss)
    assert result["en"]["loss"]==pytest.approx(expected["en"],rel=2e-6)
    assert result["zh"]["loss"]==pytest.approx(expected["zh"],rel=2e-6)
    assert result["macro_loss"]==pytest.approx((expected["en"]+expected["zh"])/2,rel=2e-6)


def test_dropout_resume_matches_continuous_and_preserves_tail_and_base(corpus):
    config,base = fixture_config(corpus,"continuous",tail=True,dropout=.2)
    base_sha = sha256_file(base)
    engine().run_sft(config)
    complete = load(Path(config["output_dir"])/"last.pt")
    assert complete["schema_version"]==1 and complete["stage"]=="sft"
    assert complete["checkpoint_kind"]=="full"
    assert {"config","model_config","model","optimizer","scheduler","scaler","rng","identity"} <= complete.keys()
    interrupted_config = deepcopy(config)
    interrupted_config["output_dir"] = str(base.parent/"interrupted")
    engine().run_sft(interrupted_config,stop_after_steps=1)
    interrupted = load(Path(interrupted_config["output_dir"])/"last.pt")
    assert interrupted["step"]==1
    engine().run_sft(interrupted_config,resume=Path(interrupted_config["output_dir"])/"last.pt")
    resumed = load(Path(interrupted_config["output_dir"])/"last.pt")
    assert complete["step"]==resumed["step"]==6
    assert complete["progress"]["examples_seen"]==resumed["progress"]["examples_seen"]==9
    assert complete["progress"]["next_update"]==resumed["progress"]["next_update"]==6
    rows=corpus[2]["train"]
    assert complete["progress"]["input_tokens_seen"]==3*sum(len(row["input_ids"])-1 for row in rows)
    assert complete["progress"]["supervised_tokens_seen"]==3*sum(row["answer_token_count"] for row in rows)
    for key in ("model","optimizer","scheduler","scaler","rng","progress"):
        assert_state_equal(complete[key],resumed[key])
    initial_model=load(base)["model"]
    assert any(not torch.equal(value,initial_model[key]) for key,value in complete["model"].items())
    assert sha256_file(base)==base_sha


def test_resume_rejects_model_only_best_checkpoint(corpus):
    config,_ = fixture_config(corpus,"reject-best")
    engine().run_sft(config,stop_after_steps=1)
    best=Path(config["output_dir"])/"best.pt"
    assert load(best)["checkpoint_kind"]=="model_only"
    with pytest.raises(ValueError):engine().run_sft(config,resume=best)


def test_resume_rejects_changed_epochs_without_overwriting_last(corpus):
    config,_ = fixture_config(corpus,"changed-plan")
    engine().run_sft(config,stop_after_steps=1)
    path=Path(config["output_dir"])/"last.pt";before=sha256_file(path)
    changed=deepcopy(config);changed["train"]["epochs"]=4.0
    with pytest.raises(ValueError):engine().run_sft(changed,resume=path)
    assert sha256_file(path)==before


def test_resume_rejects_checkpoint_copied_from_another_output(corpus):
    config,_ = fixture_config(corpus,"original-run")
    engine().run_sft(config,stop_after_steps=1)
    copied=deepcopy(config);copied["output_dir"]=str(Path(config["project_root"])/"different-run")
    output=Path(copied["output_dir"]);output.mkdir()
    checkpoint=output/"last.pt"
    checkpoint.write_bytes((Path(config["output_dir"])/"last.pt").read_bytes())
    before=sha256_file(checkpoint)
    with pytest.raises(ValueError):engine().run_sft(copied,resume=checkpoint)
    assert sha256_file(checkpoint)==before


def test_resume_rejects_new_data_identity_even_when_new_data_is_valid(corpus):
    config,_ = fixture_config(corpus,"changed-data")
    engine().run_sft(config,stop_after_steps=1)
    checkpoint=Path(config["output_dir"])/"last.pt";before=sha256_file(checkpoint)
    directory,manifest,rows,tokenizer=corpus
    row=rows["train"][0];row["response"]="你好。再见。"
    prefix=tokenizer.encode("User:\n"+row["prompt"]+"\n\nAssistant:\n",add_special_tokens=False).ids
    answer=tokenizer.encode(row["response"],add_special_tokens=False).ids+[2]
    row.update(input_ids=prefix+answer,loss_mask=[0]*len(prefix)+[1]*len(answer),prompt_token_count=len(prefix),answer_token_count=len(answer))
    rows["pilot"]=[{**deepcopy(item),"split":"pilot"} for item in rows["train"]]
    repin(directory,manifest,rows)
    changed=deepcopy(config);changed["data_manifest_sha256"]=sha256_file(directory/"manifest.json");changed["data_identity"]=manifest["data_identity"]
    with pytest.raises(ValueError):engine().run_sft(changed,resume=checkpoint)
    assert sha256_file(checkpoint)==before
