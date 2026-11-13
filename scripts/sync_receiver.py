"""SSH 同步接收端；仅依赖标准库，在远程 Python 中执行。"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time

DIRECTORIES = {"src", "configs", "requirements", "scripts", "docs", "tests", "examples"}
ROOT_FILES = {"README.md", "pyproject.toml", ".gitignore"}
SUFFIXES = {".py", ".md", ".json", ".jsonl", ".toml", ".txt", ".sh", ".ps1", ".yml", ".yaml"}
EXCLUDED = {"__pycache__", ".pytest_cache", ".venv", ".git", "_outputs", "_test_tmp", "node_modules"}
MAX_BYTES = 32 * 1024 * 1024


class SyncRefused(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def digest(data):
    return hashlib.sha256(data).hexdigest()


def allowed(name):
    """接收端独立验证白名单，不能只信任发送端。"""
    path = PurePosixPath(name)
    return (isinstance(name, str) and name == path.as_posix() and not path.is_absolute()
            and ".." not in path.parts and "\\" not in name and bool(path.parts)
            and not EXCLUDED.intersection(path.parts) and not path.name.startswith(".env")
            and (name in ROOT_FILES or (path.parts[0] in DIRECTORIES and path.suffix in SUFFIXES)))


def contained(root, name):
    path = root / name
    # 不穿过任何软链接，避免将目标写入数据目录或项目外。
    current = path
    while current != root:
        if current.is_symlink():
            raise SyncRefused("conflict", f"Symlink blocks sync: {name}")
        current = current.parent
    if not path.resolve().is_relative_to(root):
        raise SyncRefused("conflict", f"Path escapes project: {name}")
    return path


def uses_project_source(root, cwd, argv):
    if not source_python(argv):
        return False
    return cwd.is_relative_to(root) or any(b"llm_lab" in item for item in argv)


def source_python(argv):
    # pip 本身不运行训练源码；其构建子进程仍接受下面的保守检查。
    return bool(argv and b"python" in Path(os.fsdecode(argv[0])).name.lower().encode()
                and argv[1:3] != [b"-m", b"pip"])


def active_project_processes(root, proc_root=Path("/proc")):
    """保守检查项目目录中运行的 Python；不能读取时不冒险覆盖源码。"""
    found = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = (process / "cmdline").read_bytes().split(b"\0")
            if not source_python(argv):
                continue
            cwd = (process / "cwd").resolve(strict=True)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise SyncRefused("busy", "Cannot inspect remote Python processes") from exc
        if uses_project_source(root, cwd, argv):
            found.append(int(process.name))
    return found


@contextmanager
def sync_lock(directory):
    """防止两个同步进程同时应用；测试可使用同一标准接口。"""
    import fcntl
    with (directory / "apply.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def read_manifest(path):
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    files = document.get("files", {})
    if not isinstance(files, dict) or any(not allowed(name) for name in files):
        raise SyncRefused("conflict", "Invalid previous source manifest")
    return files


def apply_snapshot(root, payload, *, process_check=active_project_processes, lock=sync_lock):
    """全量小型源码快照：校验后仅替换变化文件，旧版本归档，不递归删除。"""
    root = Path(root)
    if root.is_symlink() or root.resolve() != root.absolute() or root == root.parent:
        raise SyncRefused("conflict", "Remote project must be a real absolute directory")
    if root.exists() and not root.is_dir():
        raise SyncRefused("conflict", "Remote project is not a directory")
    files, encoded = payload["files"], payload["contents"]
    if not isinstance(files, dict) or set(files) != set(encoded) or "pyproject.toml" not in files:
        raise SyncRefused("conflict", "Incomplete source snapshot")
    if any(not allowed(name) for name in files):
        raise SyncRefused("conflict", "Disallowed source path")
    contents = {name: base64.b64decode(encoded[name], validate=True) for name in files}
    if sum(map(len, contents.values())) > MAX_BYTES:
        raise SyncRefused("conflict", "Source snapshot exceeds 32 MiB limit")
    if any(digest(contents[name]) != value for name, value in files.items()):
        raise SyncRefused("conflict", "Source hash mismatch")
    if (root / ".sync-pause").exists():
        raise SyncRefused("busy", "Remote .sync-pause exists")
    busy = process_check(root)
    if busy:
        raise SyncRefused("busy", f"Project Python process is running: {busy}")
    root.mkdir(parents=True, exist_ok=True)
    directory = contained(root, ".sync")
    directory.mkdir(exist_ok=True)
    with lock(directory):
        manifest_path = contained(root, ".sync/manifest.json")
        previous = read_manifest(manifest_path)
        if not previous:
            previous = read_manifest(contained(root, "BUILD_MANIFEST.json"))
        changed, removed = {}, []
        for name in set(previous) | set(files):
            path = contained(root, name)
            if path.exists() and not path.is_file():
                raise SyncRefused("conflict", f"Source target is not a file: {name}")
            current = digest(path.read_bytes()) if path.exists() else None
            desired = files.get(name)
            if current == desired:
                continue
            if current is not None and current != previous.get(name):
                raise SyncRefused("conflict", f"Remote source has independent edits: {name}")
            if desired is None:
                if current is not None:
                    removed.append(name)
            else:
                changed[name] = contents[name]
        # 第二次检查缩小进程启动与写入之间的窗口；显式 pause 文件用于手动启动边界。
        if (root / ".sync-pause").exists() or process_check(root):
            raise SyncRefused("busy", "Project became busy before applying source")
        with tempfile.TemporaryDirectory(prefix="incoming-", dir=directory) as temporary:
            stage = Path(temporary)
            for name, data in changed.items():
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            backup = contained(root, f".sync/backups/{time.time_ns()}")
            for name in sorted(set(changed) | set(removed)):
                target = contained(root, name)
                if target.exists():
                    old = backup / name
                    old.parent.mkdir(parents=True, exist_ok=True)
                    if name in changed:
                        shutil.copy2(target, old)
                    else:
                        os.replace(target, old)
                if name in changed:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(stage / name, target)
            new_manifest = stage / "manifest.json"
            new_manifest.write_text(json.dumps({"schema_version": 1, "files": files}), encoding="utf-8")
            os.replace(new_manifest, manifest_path)
    return {"status": "synced", "changed": sorted(changed), "removed": sorted(removed), "files": len(files)}


def main():
    try:
        # stdin/argv 都来自已认证 SSH；仍限长度并检查路径，避免配置失误写错根目录。
        raw = sys.stdin.buffer.read(MAX_BYTES * 2 + 1)
        if len(raw) > MAX_BYTES * 2:
            raise SyncRefused("conflict", "Source request too large")
        payload = json.loads(raw)
        root = Path(payload["remote_root"])
        if root != Path("/root/autodl-tmp/model_training"):
            raise SyncRefused("conflict", "Unexpected remote project root")
        response = apply_snapshot(root, payload)
    except SyncRefused as exc:
        response = {"status": exc.status, "detail": str(exc)}
    except Exception as exc:
        response = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(response, ensure_ascii=True))


if __name__ == "__main__":
    main()
