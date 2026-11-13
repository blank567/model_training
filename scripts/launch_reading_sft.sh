#!/usr/bin/env bash
# 用户启动器：先检查冻结身份，再提交后台训练；终端退出后训练继续。
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf '%s\n' '用法: bash scripts/launch_reading_sft.sh runs/configs/<实验>.env [start|resume]' >&2
  exit 2
fi
reading_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$reading_root"
reading_env="$(realpath -- "$1")"
reading_action="${2:-start}"
case "$reading_env" in "$reading_root"/runs/configs/*.env) ;; *) printf '%s\n' '只允许项目内正式配置的env文件' >&2; exit 2 ;; esac
case "$reading_action" in start|resume) ;; *) printf '%s\n' 'action只能为start或resume' >&2; exit 2 ;; esac
source scripts/env_remote.sh
source "$reading_env"
if [[ "$reading_action" == start && -e "$RUN_DIR/last.pt" ]]; then
  printf '%s\n' '该实验已有断点；请将第二个参数设为resume，继续原定一轮。' >&2
  exit 1
fi
if [[ "$reading_action" == resume && ! -f "$RUN_DIR/last.pt" ]]; then
  printf '%s\n' '该实验没有last.pt，不能恢复。' >&2
  exit 1
fi
printf '%s\n' '正在检查冻结配置、数据和起始权重，请等待检查结果。'
.venv/bin/python -u scripts/train_reading_sft.py check --config "$RUN_CONFIG"
mkdir -p -- "$(dirname -- "$RUN_LOG")"
nohup .venv/bin/python -u scripts/train_reading_sft.py "$reading_action" --config "$RUN_CONFIG" >>"$RUN_LOG" 2>&1 < /dev/null &
reading_pid=$!
printf '%s\n' "$reading_pid" > "${RUN_LOG%.log}.pid"
printf '后台启动已提交，PID=%s\n日志: %s\n' "$reading_pid" "$RUN_LOG"
printf '查看进度: tail -f %q\n' "$RUN_LOG"
