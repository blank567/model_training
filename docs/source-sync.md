# 本地保存后自动同步到 AutoDL

本地 `E:\codex\model_training` 是源码主版本。同步器每秒计算一次源码哈希，连续保存稳定约 1 秒后，通过 SSH 发送当前源码快照；通常保存后数秒内更新远程。它不会登录远程编辑代码，也不会启动训练、安装依赖或重启进程。

目标固定为 `/root/autodl-tmp/model_training`。连接读取用户自己的 SSH config；专用私钥、已固定的主机公钥、缓存、控制文件和日志均在本项目 `tmp/` 下。首次使用复制 `configs/deployment/sync_autodl.example.json` 为忽略的 `sync_autodl.json`，填写自己的连接信息和 SSH config 路径。非交互 SSH 使用明确的 `/root/miniconda3/bin/python`，不依赖终端加载 Conda。

## 控制命令

在本机 PowerShell 执行；默认使用现有 `E:\conda_envs\llm\python.exe`，没有增加第三方依赖：

```powershell
& E:\codex\model_training\scripts\sync.ps1 Start
& E:\codex\model_training\scripts\sync.ps1 Status
& E:\codex\model_training\scripts\sync.ps1 Pause
& E:\codex\model_training\scripts\sync.ps1 Resume
& E:\codex\model_training\scripts\sync.ps1 Stop
```

本机若禁止直接执行 `.ps1`，可使用仅对当前启动进程生效的命令，不改变系统策略：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File E:\codex\model_training\scripts\sync.ps1 Status
```

`Start` 在隐藏窗口的后台进程中持续运行；重复执行不会创建多个同步器。关闭 VS Code 不会主动停止同步器，Windows 退出登录或重启后需要再次执行 `Start`。没有添加系统开机任务。修改同步器脚本或连接配置后先 `Stop`，等待状态显示停止，再 `Start`。

前台单次同步用于排错：

```powershell
& E:\codex\model_training\scripts\sync.ps1 Stop
# 等待 Status 中 running=false 后：
& E:\codex\model_training\scripts\sync.ps1 Once
```

## 范围和写入规则

白名单包括 `src/`、`configs/`、`requirements/`、`scripts/`、`docs/`、`tests/`、必要 `examples/` 及 README、pyproject、gitignore。缓存、环境、`tmp/ssh`、正式数据、权重、训练日志和 Checkpoint 不参与同步；`.env` 文件和 fixture 产物也被排除。VS Code 远程编辑不会回传到本地，本工具是单向同步。

接收端验证路径、软链接和 SHA256，再暂存文件、逐文件原子替换。首次同步可接续源码包的 `BUILD_MANIFEST.json`；如果既有文件不属于可验证的已同步版本且与本地不同，则报告 `conflict`，不覆盖该批次。后续发现远程独立修改也会拒绝覆盖，需要把需要的改动带回本地主版本后再同步。

本地删除或重命名时，只将上次 manifest 管理过的对应远程源码移到 `.sync/backups/<批次>/`；未管理的远程文件和训练产物保留。源码旧版本也归档至该目录，没有递归镜像删除。备份暂不自动清理。

## 训练期间暂停

远程接收端发现项目目录中的 Python 进程，或任意 `llm_lab` Python 任务时，返回 `busy`，每 10 秒重试；`python -m pip` 安装进程除外，其构建子进程仍检查。数据处理、测试、训练等任务完成后会同步最新本地版本，不会把中间保存逐次回放。

进程扫描无法完全消除启动任务与写入之间的竞争，也不覆盖一切自定义启动方式。正式训练前先确认当前快照已 `synced`，在本机执行 `Pause`，再在远程创建显式暂停标记：

```bash
cd /root/autodl-tmp/model_training
touch .sync-pause
```

运行结束后移除该标记，再在本机 `Resume`；此后才发布积累的源码修改。断点恢复仍要求原代码/配置/数据身份一致，不能用同步新代码代替兼容性检查。

## 状态与故障

`tmp/sync/status.json` 记录 PID、状态、最后成功时间、源码快照身份；`events.jsonl` 记录传输结果，`stdout.log`/`stderr.log` 用于后台排错。状态为 `synced` 才表示远程接收端确认过快照。`pending`/`syncing` 表示等待或传输；`busy` 表示延迟；`authentication_required`/`connection_error` 会自动重试；`conflict` 需要检查具体路径。

断线、GPU 实例关机或 SSH 认证失败时，本地变更不会被确认为成功；连接恢复后发送最新完整小型源码快照。实例关机后若主机/端口变化，更新连接配置并重启同步器。主机密钥变更会拒绝连接，需要先核验，不能自动忽略。

同步不启动 GPU 运算，但实例开机等待仍按平台规则计费。这里的自动化只涉及源码传输，不会自动关机或释放实例。

## 2026-10-23 实际验收

后台同步器已在本机启动，并通过真实 AutoDL SSH 验证新增、修改、重命名、删除，单次观测分别约 2.812、2.813、2.844、2.312 秒。该数值包含轮询、稳定等待、SSH 和校验，不保证所有网络条件下延迟相同。

另用项目目录下一个 8 秒的 Python 等待进程验证：进程存活时不发布变更，退出后自动补齐。没有为验收运行 GPU 训练。本机全套测试为 99 passed、1 skipped；跳过项仍是目标 PEFT 环境的真实 adapter 恢复测试。

验收细节保存在本机 `tmp/sync/integration_report.json` 和 `tmp/sync/safety_report.json`，实时状态以 `tmp/sync/status.json` 和 `Status` 命令为准。源码同步完成不代表训练依赖、CUDA 或 BF16 已验收。
