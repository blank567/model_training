# AutoDL W&B网络修复：2026-11-01

已完成服务器认证、SDK在线写入、云端读回，以及四组LoRA离线记录补传。没有重新训练模型，没有修改历史冻结配置、Checkpoint或数值训练源码。

## 实测原因与修复路径

同一W&B API：本机直连与AutoDL直连均返回HTTP403；AutoDL自带加速线路也返回403。本机现有127.0.0.1:7890代理可正常连接。复用该网络出口后，AutoDL GraphQL认证返回HTTP200，原有API Key有效；账号不需要重新登录。

链路为：AutoDL SDK或同步CLI → 服务器127.0.0.1:17890 → 专用SSH反向隧道 → 本机127.0.0.1:17891的W&B中继 → 本机现有127.0.0.1:7890代理 → W&B。

本机中继只接受W&B域名和所需存储域名的443端口CONNECT，不解密TLS、不记录请求内容。服务器监听经/proc验证为0100007F:45E2，即127.0.0.1:17890，没有对外开放通用代理。原控制密钥的restrict保持不变；另建的隧道专用密钥禁用远程命令，TCP反向监听限定127.0.0.1:17890。实际验证了远程命令不可执行、其他TCP监听端口被拒绝。

API Key仍只位于服务器已有NETRC，没有复制到本机，没有写入聊天或网络配置。私钥和状态文件均位于E盘tmp/wandb_network，不进入源码包。

## 当前补传结果

| 实验 | W&B |
| --- | --- |
| r8 / alpha16 / dropout0 | [原始运行](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/a3422e345dab) |
| r32 / alpha64 / dropout0 | [原始运行](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/ca000af78489) |
| r32 / alpha64 / dropout0.05 | [原始运行](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/db047cc280e2) |
| r32 / alpha64 / dropout0.1 | [原始运行](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/77c20211b8d7) |

每组本地315条历史记录与云端315条逐条optimizer_step序列相同，最终步数3125，云端状态finished。保留原run ID；没有把同步CLI的退出码0当作唯一成功依据。cloud_sync.json单独记录上传结果，未改原wandb_run.json中的offline身份。scripts/lora_ablation_status.py会显示已补传的链接。

一次纯CPU在线探针也已写入并读回：[网络验证运行](https://wandb.ai/mblankm-peking-university/bilingual-training-lab/runs/onelf2mw)。它标有assistant/network-check，只有连通性数值，没有模型训练。

## 日常启动与检查

本机PowerShell：

```powershell
cd E:\codex\model_training
.\scripts\wandb_network.ps1 Start
.\scripts\wandb_network.ps1 Status
```

Start以隐藏窗口方式启动本机后台中继与SSH守护。SSH断线时会自动重连。日志在runs/assistant/system/wandb_network_20261101_172809，状态在tmp/wandb_network/status.json。

AutoDL远程Bash：

```bash
cd /root/autodl-tmp/model_training
source scripts/env_remote.sh
.venv/bin/python scripts/check_wandb_connection.py
```

返回credential_verified=true代表服务器真正验证了W&B凭据。可用--write-test额外做一次在线写入和云端读回；每次会产生一个标注为连通性检查的运行，不使用GPU。

手动上传指定离线目录：

```bash
bash scripts/wandb_online.sh sync --yes --include-synced \
  --entity mblankm-peking-university --project bilingual-training-lab \
  /完整路径/wandb/offline-run-...
```

同步CLI只在当前进程使用HTTP代理。训练SDK通过env_remote.sh加载.cache/wandb/network.env里的WANDB_HTTP_PROXY和WANDB_HTTPS_PROXY，其他数据下载不被强制改走这条中继。NO_PROXY包含本机地址且可重复source。

两个未冻结的单轮LoRA模板已恢复wandb_mode=online；历史runs/configs配置继续保留offline，通过补传在云端可见。后续仍是epochs=1、stop_epochs=null。不要为了切换W&B模式编辑已冻结配置后resume。

## 连接依赖与停止

**此方案依赖本机开机、现有7890代理运行，以及SSH隧道运行。** Windows重启或注销后，需要再执行Start。AutoDL关机、SSH地址/端口改变或代理停止时也会断开；恢复原地址和代理后，守护会重试。它不是服务器脱离本机后的独立代理服务。

```powershell
.\scripts\wandb_network.ps1 Stop
```

Stop只请求本机网络守护退出，不向任何模型训练进程发送终止信号。在线训练是否能够启动，仍取决于当时网络是否可达；连接失败时先检查Status和服务器检查命令，不要反复覆盖API Key。

## 验证记录

12项CPU/回环检查通过，覆盖允许域名、拒绝任意目标/非TLS端口、TLS字节透传与云端历史缺失拒绝。服务器真实认证、SDK在线写入与读回通过，四组完整历史均核对通过。原Full与LoRA数值源码指纹分别保持0890bc473d14ed726362a6a4250d809c9389430f7786ab44fd41b8a021cd3872与899d9750e6b9963cf0e5f60f22a3faf51655780d9519f919396da50fb0d3f914。

远程完整补传报告：runs/assistant/system/wandb_backfill_20261101_173528/report.json。新增隧道公钥的选项与原文件备份在runs/assistant/system/wandb_network_20261101_172809；本机私钥未上传。
