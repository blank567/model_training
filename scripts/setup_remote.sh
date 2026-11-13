#!/usr/bin/env bash
# 默认只校验；--install 才安装。不会安装驱动或修改系统 CUDA Toolkit。
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_remote.sh"
cd -- "$LLM_LAB_ROOT"

cat <<'NOTICE'
环境边界：NVIDIA 驱动由云平台提供；PyTorch CUDA wheel 自带所需用户态运行库；
系统 CUDA Toolkit（nvcc）用于编译开发，三者版本含义不同。
本脚本不会修改驱动或系统 Toolkit。安装 wheel 可能下载数 GB 文件，
仅在显式提供 --install 时执行；建议在新的项目 .venv 中安装。
NOTICE

if [[ "${1:-}" != "" && "${1:-}" != "--install" ]]; then
  printf '%s\n' '用法：bash scripts/setup_remote.sh [--install]' >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "${1:-}" != "--install" && -x "$LLM_LAB_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$LLM_LAB_ROOT/.venv/bin/python"
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] in {(3, 10), (3, 11)}, "请使用 Python 3.10 或 3.11"'

if [[ "${1:-}" == "--install" ]]; then
  "$PYTHON_BIN" -m venv "$LLM_LAB_ROOT/.venv"
  PYTHON_BIN="$LLM_LAB_ROOT/.venv/bin/python"
  "$PYTHON_BIN" -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
  "$PYTHON_BIN" -m pip install -r requirements/remote.txt
  "$PYTHON_BIN" -m pip install -e '.[posttrain,dev]'
fi

"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -m pip freeze --all > "$LLM_LAB_ROOT/runs/pip-freeze.txt"
"$PYTHON_BIN" -m llm_lab.runtime.probe --device cuda --require-cuda --output runs/environment.json
printf '%s\n' 'GPU 小矩阵校验通过。接下来仍需真实模型的短程训练测显存和 tokens/s。'
printf '%s\n' '训练进程退出不会释放计费实例；停止/释放实例需要在云平台另行操作。'
