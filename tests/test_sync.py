"""验证源码同步不会覆盖远程产物、独立编辑或运行中的项目。"""
import base64
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))

    def load(name):
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load("sync_project"), load("sync_receiver")


def source(root, **files):
    root.mkdir(parents=True, exist_ok=True)
    for name, text in {"pyproject.toml": "[project]\nname='sync-fixture'\n", **files}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def apply(receiver, root, snapshot, **kwargs):
    return receiver.apply_snapshot(root, snapshot, process_check=lambda root: [], lock=lambda _: nullcontext(), **kwargs)


def config():
    return {"remote_root": "/root/autodl-tmp/model_training", "poll_seconds": 1,
            "debounce_seconds": 1, "retry_seconds": 10, "timeout_seconds": 30,
            "host": "example.test", "port": 17712, "user": "root", "ssh_config": "E:/config",
            "identity_file": "tmp/ssh/task-key", "known_hosts_file": "tmp/ssh/known_hosts",
            "remote_python": "/root/miniconda3/bin/python"}


def test_snapshot_excludes_credentials_outputs_and_weights(modules, tmp_path):
    local, _ = modules
    source(tmp_path, **{"src/code.py": "print(1)", "configs/test.json": "{}", "scripts/.env.secret": "secret",
                        "tmp/ssh/key.json": "private", "runs/config.json": "output",
                        "examples/_outputs/file.json": "output", "src/weights.pt": "weights"})
    assert set(local.snapshot(tmp_path)["files"]) == {"pyproject.toml", "src/code.py", "configs/test.json"}


def test_create_update_rename_archives_only_managed_source(modules, tmp_path):
    local, remote = modules
    origin, target = tmp_path / "local", tmp_path / "remote"
    source(origin, **{"src/old.py": "value=1"})
    first = apply(remote, target, local.snapshot(origin))
    assert first["status"] == "synced"
    artifact = target / "runs/last.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"checkpoint")
    (target / "src/unmanaged.py").write_text("preserve")
    (origin / "src/old.py").rename(origin / "src/new.py")
    (origin / "src/new.py").write_text("value=2")
    second = apply(remote, target, local.snapshot(origin))
    assert second["removed"] == ["src/old.py"]
    assert not (target / "src/old.py").exists()
    assert (target / "src/new.py").read_text() == "value=2"
    assert list((target / ".sync/backups").glob("*/src/old.py"))[0].read_text() == "value=1"
    assert artifact.read_bytes() == b"checkpoint"
    assert (target / "src/unmanaged.py").read_text() == "preserve"
    assert apply(remote, target, local.snapshot(origin))["changed"] == []


def test_remote_edits_refuse_entire_batch_before_writes(modules, tmp_path):
    local, remote = modules
    origin, target = tmp_path / "local", tmp_path / "remote"
    source(origin, **{"src/a.py": "original", "src/b.py": "original"})
    apply(remote, target, local.snapshot(origin))
    (target / "src/a.py").write_text("remote-edit")
    (origin / "src/b.py").write_text("local-update")
    with pytest.raises(remote.SyncRefused, match="independent edits"):
        apply(remote, target, local.snapshot(origin))
    assert (target / "src/a.py").read_text() == "remote-edit"
    assert (target / "src/b.py").read_text() == "original"


def test_first_sync_accepts_verified_source_bundle_baseline(modules, tmp_path):
    local, remote = modules
    origin, target = tmp_path / "local", tmp_path / "remote"
    source(origin, **{"src/a.py": "old"})
    source(target, **{"src/a.py": "old"})
    baseline = local.snapshot(origin)
    (target / "BUILD_MANIFEST.json").write_text(json.dumps({"files": baseline["files"]}))
    (origin / "src/a.py").write_text("new")
    apply(remote, target, local.snapshot(origin))
    assert (target / "src/a.py").read_text() == "new"


@pytest.mark.parametrize("name", ["../escape.py", "/tmp/escape.py", "src/../../escape.py", "runs/config.json", "scripts/.env.secret", "src\\escape.py"])
def test_receiver_rejects_unsafe_source_paths(modules, tmp_path, name):
    local, remote = modules
    source(tmp_path / "local")
    request = local.snapshot(tmp_path / "local")
    request["files"][name] = hashlib.sha256(b"data").hexdigest()
    request["contents"][name] = base64.b64encode(b"data").decode()
    with pytest.raises(remote.SyncRefused, match="Disallowed"):
        apply(remote, tmp_path / "remote", request)
    assert not (tmp_path / "remote").exists()


def test_hash_corruption_rejected_before_remote_creation(modules, tmp_path):
    local, remote = modules
    source(tmp_path / "local")
    request = local.snapshot(tmp_path / "local")
    request["contents"]["pyproject.toml"] = base64.b64encode(b"corruption").decode()
    with pytest.raises(remote.SyncRefused, match="hash mismatch"):
        apply(remote, tmp_path / "remote", request)
    assert not (tmp_path / "remote").exists()


def test_receiver_refuses_symlink_parent(modules, tmp_path, monkeypatch):
    local, remote = modules
    origin, target = tmp_path / "local", tmp_path / "remote"
    source(origin, **{"src/a.py": "new"})
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == target / "src" or original(path))
    with pytest.raises(remote.SyncRefused, match="Symlink"):
        apply(remote, target, local.snapshot(origin))
    assert not (target / "src/a.py").exists()


def test_busy_project_defers_and_pause_marker_blocks(modules, tmp_path):
    local, remote = modules
    source(tmp_path / "local")
    target = tmp_path / "remote"
    request = local.snapshot(tmp_path / "local")
    with pytest.raises(remote.SyncRefused, match="running"):
        remote.apply_snapshot(target, request, process_check=lambda root: [123], lock=lambda _: nullcontext())
    assert not target.exists()
    target.mkdir()
    (target / ".sync-pause").touch()
    with pytest.raises(remote.SyncRefused, match="pause"):
        apply(remote, target, request)


def test_busy_on_second_check_still_preserves_original_source(modules, tmp_path):
    local, remote = modules
    source(tmp_path / "local", **{"src/a.py": "new"})
    count = iter([[], [123]])
    with pytest.raises(remote.SyncRefused, match="became busy"):
        remote.apply_snapshot(tmp_path / "remote", local.snapshot(tmp_path / "local"),
                              process_check=lambda root: next(count), lock=lambda _: nullcontext())
    assert not (tmp_path / "remote/pyproject.toml").exists()


def test_watcher_debounces_and_retries_latest_content_after_failure(modules, tmp_path):
    local, _ = modules
    origin, state = tmp_path / "local", tmp_path / "state"
    source(origin, **{"src/a.py": "aaa"})
    state.mkdir()
    now, sent = [0], []

    def transport(cfg, request):
        sent.append(request)
        return {"status": "connection_error" if len(sent) == 1 else "synced"}

    watcher = local.Watcher(origin, config(), state, transport=transport, clock=lambda: now[0])
    assert watcher.tick()["status"] == "pending"
    now[0] = 2
    assert watcher.tick()["status"] == "connection_error"
    assert watcher.acknowledged is None
    (origin / "src/a.py").write_text("bbb")  # 相同文件大小的保存同样被哈希识别。
    now[0] = 4
    watcher.tick()
    assert len(sent) == 1
    now[0] = 13
    assert watcher.tick()["status"] == "synced"
    assert base64.b64decode(sent[-1]["contents"]["src/a.py"]) == b"bbb"
    now[0] = 15
    watcher.tick()
    assert len(sent) == 2
    assert watcher.acknowledged == sent[-1]["identity"]


def test_local_pause_retains_pending_changes(modules, tmp_path):
    local, _ = modules
    source(tmp_path / "local")
    state = tmp_path / "state"
    state.mkdir()
    (state / "pause").touch()
    sent = []
    watcher = local.Watcher(tmp_path / "local", config(), state,
                            transport=lambda cfg, current: sent.append(current) or {"status": "synced"})
    assert watcher.tick(force=True)["status"] == "paused"
    assert not sent
    (state / "pause").unlink()
    assert watcher.tick(force=True)["status"] == "synced"


def test_transport_uses_pinned_host_noninteractive_and_preserves_auth_failure(modules, monkeypatch):
    local, _ = modules
    monkeypatch.setattr(local.shutil, "which", lambda name: "ssh.exe")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=255, stderr=b"Permission denied (publickey,password).", stdout=b"")

    result = local.transfer(config(), {"files": {}, "contents": {}}, run=run)
    assert result["status"] == "authentication_required"
    command, options = calls[0]
    assert "BatchMode=yes" in command and "StrictHostKeyChecking=yes" in command
    assert "shell" not in options and command[-2] == "root@example.test"


def test_singleton_prevents_duplicate_watcher(modules, tmp_path):
    local, _ = modules
    with local.singleton(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with local.singleton(tmp_path):
                pytest.fail("second watcher acquired lock")


def test_failed_atomic_replacement_keeps_original_and_can_retry(modules, tmp_path, monkeypatch):
    local, remote = modules
    origin, target = tmp_path / "local", tmp_path / "remote"
    source(origin, **{"src/a.py": "old"})
    apply(remote, target, local.snapshot(origin))
    (origin / "src/a.py").write_text("new")
    replace = remote.os.replace

    def fail_source_replacement(incoming, destination):
        if Path(destination) == target / "src/a.py":
            raise OSError("simulated replacement failure")
        replace(incoming, destination)

    with monkeypatch.context() as patch:
        patch.setattr(remote.os, "replace", fail_source_replacement)
        with pytest.raises(OSError, match="simulated"):
            apply(remote, target, local.snapshot(origin))
    assert (target / "src/a.py").read_text() == "old"
    apply(remote, target, local.snapshot(origin))
    assert (target / "src/a.py").read_text() == "new"


@pytest.mark.parametrize("argv,inside,expected", [
    ([b"python", b"-m", b"pip", b"install", b"torch"], True, False),
    ([b"python", b"-m", b"pytest"], True, True),
    ([b"python", b"custom_training.py"], True, True),
    ([b"python", b"-m", b"llm_lab.cli", b"pretrain"], False, True),
    ([b"node", b"server.js"], True, False),
])
def test_source_process_detection_distinguishes_pip_from_training(modules, tmp_path, argv, inside, expected):
    _, remote = modules
    root = tmp_path / "project"
    assert remote.uses_project_source(root, root if inside else tmp_path, argv) is expected


@pytest.mark.parametrize("executable", [b"sshd", b"python"])
def test_unreadable_nonpython_cwd_is_ignored_but_python_stays_protected(modules, tmp_path, monkeypatch, executable):
    _, remote = modules
    proc = tmp_path / "proc"
    process = proc / "123"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(executable + b"\0")
    resolve = Path.resolve

    def denied_cwd(path, *args, **kwargs):
        if path == process / "cwd":
            raise PermissionError("container process metadata restriction")
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", denied_cwd)
    if executable == b"python":
        with pytest.raises(remote.SyncRefused, match="Cannot inspect"):
            remote.active_project_processes(tmp_path / "project", proc)
    else:
        assert remote.active_project_processes(tmp_path / "project", proc) == []
