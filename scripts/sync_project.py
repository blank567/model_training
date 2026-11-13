"""本地源码单向实时同步；标准库轮询 + 已固定主机密钥的 OpenSSH。"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from package_project import source_files

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "tmp/sync"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def snapshot(root):
    files, contents = {}, {}
    total = 0
    for path in source_files(root):
        if path.stat().st_size > 32 * 1024 * 1024 - total:
            raise ValueError("Source snapshot exceeds 32 MiB limit")
        data = path.read_bytes()
        total += len(data)
        if total > 32 * 1024 * 1024:
            raise ValueError("Source snapshot exceeds 32 MiB limit")
        name = path.relative_to(root).as_posix()
        files[name] = hashlib.sha256(data).hexdigest()
        contents[name] = base64.b64encode(data).decode("ascii")
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"identity": identity, "files": files, "contents": contents}


def load_config(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["schema_version"] != 1 or cfg["remote_root"] != "/root/autodl-tmp/model_training":
        raise ValueError("Unexpected sync schema or target root")
    if not isinstance(cfg["port"], int) or not 1 <= cfg["port"] <= 65535:
        raise ValueError("Invalid SSH port")
    for name in ("poll_seconds", "debounce_seconds", "retry_seconds", "timeout_seconds"):
        if not isinstance(cfg[name], (int, float)) or not 0.1 <= cfg[name] <= 300:
            raise ValueError(f"Invalid {name}")
    return cfg


def transfer(cfg, current, *, run=subprocess.run):
    """参数数组避免 Windows shell 展开；远程命令使用 POSIX quoting。"""
    ssh = shutil.which("ssh")
    if not ssh:
        raise RuntimeError("OpenSSH ssh executable not found")
    resolve = lambda name: str((ROOT / cfg[name]).resolve())
    command = [ssh, "-F", cfg["ssh_config"], "-p", str(cfg["port"]), "-i", resolve("identity_file"),
               "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=8",
               "-o", "ConnectionAttempts=1", "-o", "StrictHostKeyChecking=yes",
               "-o", "UpdateHostKeys=no", "-o", f"UserKnownHostsFile={resolve('known_hosts_file')}"]
    receiver = base64.b64encode((ROOT / "scripts/sync_receiver.py").read_bytes()).decode("ascii")
    program = f"import base64;exec(compile(base64.b64decode('{receiver}'),'<sync_receiver>','exec'))"
    command += [f"{cfg['user']}@{cfg['host']}", shlex.join([cfg["remote_python"], "-c", program])]
    request = {"remote_root": cfg["remote_root"], "files": current["files"], "contents": current["contents"]}
    result = run(command, input=json.dumps(request).encode(), capture_output=True, timeout=cfg["timeout_seconds"])
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        return {"status": "authentication_required" if "Permission denied" in error else "connection_error",
                "detail": error[-2000:] or f"ssh exit code {result.returncode}"}
    response = json.loads(result.stdout.decode("utf-8"))
    if response.get("status") not in {"synced", "busy", "conflict", "error"}:
        raise RuntimeError("Unexpected sync receiver response")
    return response


@contextmanager
def singleton(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "watcher.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Sync watcher is already running") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def running():
    try:
        with singleton(STATE):
            return False
    except RuntimeError:
        return True


class Watcher:
    """失败不确认快照；新变更保留到下一次成功，保证断线后可收敛。"""
    def __init__(self, root, cfg, state, *, transport=transfer, clock=time.monotonic):
        self.root, self.cfg, self.state = root, cfg, state
        self.transport, self.clock = transport, clock
        self.observed = None
        self.acknowledged = None
        self.changed_at = 0.0
        self.retry_at = 0.0
        self.last_success = None
        self.last_response = {"status": "starting"}

    def status(self, **updates):
        record = {"pid": os.getpid(), "updated_at_utc": timestamp(), "local_root": str(self.root),
                  "remote_root": self.cfg["remote_root"], "last_success_utc": self.last_success,
                  "acknowledged_identity": self.acknowledged, **self.last_response, **updates}
        write_json(self.state / "status.json", record)
        return record

    def tick(self, *, force=False):
        current = snapshot(self.root)
        now = self.clock()
        if current["identity"] != self.observed:
            self.observed, self.changed_at = current["identity"], now
        if (self.state / "pause").exists():
            return self.status(status="paused")
        if not force and (current["identity"] == self.acknowledged or now < self.retry_at):
            return self.status()
        if not force and now - self.changed_at < self.cfg["debounce_seconds"]:
            return self.status(status="pending", tracked_files=len(current["files"]))
        self.status(status="syncing", tracked_files=len(current["files"]))
        try:
            response = self.transport(self.cfg, current)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            response = {"status": "connection_error", "detail": str(exc)}
        self.last_response = response
        if response["status"] == "synced":
            self.acknowledged, self.last_success = current["identity"], timestamp()
            self.retry_at = 0.0
        else:
            self.retry_at = self.clock() + self.cfg["retry_seconds"]
        event = self.status(tracked_files=len(current["files"]))
        with (self.state / "events.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=True), flush=True)
        return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deployment/sync_autodl.json")
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ("watch", "once", "status", "stop", "pause", "resume"):
        actions.add_argument("--" + action, action="store_true")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    if args.status:
        document = json.loads((STATE / "status.json").read_text(encoding="utf-8")) if (STATE / "status.json").exists() else {}
        active = running()
        print(json.dumps({**document, "running": active}, ensure_ascii=True, indent=2))
        return 0 if active else 1
    if args.stop or args.pause:
        (STATE / ("stop" if args.stop else "pause")).touch()
        print("Stop requested" if args.stop else "Sync paused")
        return 0
    if args.resume:
        (STATE / "pause").unlink(missing_ok=True)
        print("Sync resumed")
        return 0
    cfg = load_config(args.config.resolve())
    with singleton(STATE):
        (STATE / "stop").unlink(missing_ok=True)
        watcher = Watcher(ROOT, cfg, STATE)
        if args.once:
            return 0 if watcher.tick(force=True)["status"] == "synced" else 1
        watcher.status()
        try:
            while not (STATE / "stop").exists():
                try:
                    watcher.tick()
                except (OSError, ValueError) as exc:
                    watcher.last_response = {"status": "local_error", "detail": str(exc)}
                    watcher.status()
                time.sleep(cfg["poll_seconds"])
        finally:
            watcher.status(status="stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
