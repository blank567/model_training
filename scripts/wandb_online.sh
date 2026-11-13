#!/usr/bin/env bash
# 只给W&B CLI设置通用HTTP代理，训练SDK使用WANDB_HTTP(S)_PROXY。
set -euo pipefail
LLM_WANDB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LLM_WANDB_ROOT"
source scripts/env_remote.sh
if [[ -n "${WANDB_HTTPS_PROXY:-}" ]]; then
  export HTTPS_PROXY="$WANDB_HTTPS_PROXY" https_proxy="$WANDB_HTTPS_PROXY"
  export HTTP_PROXY="${WANDB_HTTP_PROXY:-$WANDB_HTTPS_PROXY}" http_proxy="${WANDB_HTTP_PROXY:-$WANDB_HTTPS_PROXY}"
fi
exec .venv/bin/wandb "$@"
