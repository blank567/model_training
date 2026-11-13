"""生成可上传的源码包，明确排除缓存、权重、运行产物和凭据。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

DIRECTORIES = ("src", "configs", "requirements", "scripts", "docs", "tests", "examples")
ROOT_FILES = ("README.md", "pyproject.toml", ".gitignore")
SUFFIXES = {".py", ".md", ".json", ".jsonl", ".toml", ".txt", ".sh", ".ps1", ".yml", ".yaml"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git", "_outputs", "_test_tmp", "node_modules"}


def source_files(root: Path) -> list[Path]:
    result = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in DIRECTORIES:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if (not path.is_file() or path.is_symlink() or EXCLUDED_PARTS.intersection(relative.parts)
                    or path.suffix not in SUFFIXES or path.name.startswith(".env")
                    or not path.resolve().is_relative_to(root)):
                continue
            result.append(path)
    return sorted(result, key=lambda value: value.relative_to(root).as_posix())


def build_archive(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("Archive must stay inside the project workspace")
    files = source_files(root)
    identities = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest = {"schema_version": 1, "kind": "source_only_deployment_bundle", "files": identities}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name, content in [("BUILD_MANIFEST.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))] + [
                    (path.relative_to(root).as_posix(), path.read_bytes()) for path in files]:
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(content), 0, 0o644
                archive.addfile(info, io.BytesIO(content))
    temporary.replace(output)
    return {"archive": str(output), "files": len(files), "size_bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/model_training_autodl.tar.gz")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(build_archive(root, root / args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
