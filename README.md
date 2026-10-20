# Coding Agent

候选训练数据原始快照位于 [data/training/](data/training/README.md)，与评测数据分开存放；固定版本、下载状态和校验结果见该目录的 `manifest.json`。

评测体系统一位于 [benchmark/](benchmark/README.md)：四套数据、官方评测器和脚本、Harbor、Apptainer、依赖环境、版本清单与评测日志均集中在该目录。

```bash
cd /mnt/data/rui.yan/projects/coding_agent/benchmark
bash scripts/manage_tb2.sh harbor --version
bash scripts/verify_data.sh
```

各评测集的当前进度和结果、运行命令，以及 Terminal-Bench 各轮结果目录的索引见 [benchmark/README.md](benchmark/README.md)。

## 沙盒控制

[sandbox/](sandbox/README.md) 存放 ACP CPU Worker 内的 Apptainer 沙盒及其外部 HTTP 控制服务，供 Agent RL 远程执行命令、读写文件和使用终端。目前只完成单沙盒执行层，调度层和 64 路并发尚未实现。

## 模型服务

当前模型服务独立管理，记录见 `logs/current-service.json`，配置见 `configs/qwen36-35b-a3b.json`。评测使用的模型接口信息归档在 `benchmark/archive/model-endpoint.json`，调用时显式传入服务地址和模型名。

新增三个四卡副本使用 `qwen36-35b-r1/r2/r3` 作业名前缀，部署和就绪状态记录在 `logs/qwen36-35b-replicas.json`。已完成的 Base HLE 同时使用原 `logs/current-service.json` 记录的第一台服务和这三个新副本，每台 32 并发，总计 128。

Qwen3.6-35B-A3B 权重 `/mnt/model/ckpts/Qwen3.6-35B-A3B`，使用 `/mnt/data/rui.yan/venvs/vllm`，4 张 A800-80GB、TP=4、端口 8200、262144 上下文；启用 MTP、`qwen3` reasoning parser 与 `qwen3_xml` tool parser。

已分配 GPU 的 Worker 中可运行：

```bash
bash /mnt/data/rui.yan/projects/coding_agent/run_vllm_serve.sh
```

本次目录整理保留正在运行的模型服务及其脚本、配置和服务日志位置。
