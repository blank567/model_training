#!/usr/bin/env bash
# 用 source 加载；所有项目依赖缓存和临时文件留在所选数据盘的项目目录。
LLM_LAB_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLM_LAB_ROOT="${LLM_LAB_ROOT:-$(cd -- "$LLM_LAB_SCRIPT_DIR/.." && pwd)}"
if [[ ! -d "$LLM_LAB_ROOT" ]]; then
  printf '%s\n' 'LLM_LAB_ROOT 必须是已存在的项目目录。' >&2
  return 1 2>/dev/null || exit 1
fi
export LLM_LAB_ROOT="$(cd -- "$LLM_LAB_ROOT" && pwd -P)"
if [[ ! -w "$LLM_LAB_ROOT" || "$LLM_LAB_ROOT" == "/" || ! -f "$LLM_LAB_ROOT/pyproject.toml" || ! -d "$LLM_LAB_ROOT/src/llm_lab" ]]; then
  printf '%s\n' 'LLM_LAB_ROOT 必须是可写的本项目根目录。' >&2
  return 1 2>/dev/null || exit 1
fi
export HF_HOME="$LLM_LAB_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR="$LLM_LAB_ROOT/.cache/pip"
export TORCH_HOME="$LLM_LAB_ROOT/.cache/torch"
export XDG_CACHE_HOME="$LLM_LAB_ROOT/.cache"
export TMPDIR="$LLM_LAB_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
unset WANDB_MODE WANDB_DISABLED
export WANDB_DIR="$LLM_LAB_ROOT/runs/wandb"
export WANDB_CONFIG_DIR="$LLM_LAB_ROOT/.cache/wandb/config"
export WANDB_CACHE_DIR="$LLM_LAB_ROOT/.cache/wandb/cache"
export WANDB_DATA_DIR="$LLM_LAB_ROOT/.cache/wandb/staging"
export WANDB_ARTIFACT_DIR="$LLM_LAB_ROOT/artifacts/wandb"
export WANDB_CREDENTIALS_FILE="$LLM_LAB_ROOT/.cache/wandb/credentials.json"
export NETRC="$LLM_LAB_ROOT/.cache/wandb/netrc"
export WANDB_DISABLE_CODE=true
export WANDB_LOG_MODEL=false
export WANDB_WATCH=false
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$LLM_LAB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p -- "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TMPDIR" "$LLM_LAB_ROOT/runs"
mkdir -p -- "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"
# W&B专用网络设置与密钥分开保存；不把代理强加给其他数据下载工具。
if [[ -f "$LLM_LAB_ROOT/.cache/wandb/network.env" ]]; then
  source "$LLM_LAB_ROOT/.cache/wandb/network.env"
fi
