# 项目完成情况

更新日期：2026-10-21

## 结论

本仓库围绕 Qwen3.6-35B-A3B 的 coding agent，完成了评测体系、沙盒控制系统、SFT 数据与训练流水线三套基础设施，并产出了 HLE 全量、TB2 和 ASI-Bench 的部分评测结果。SFT 训练任务已经提交，但因 GPU 配额不足一直停留在 STARTING，尚未真正开始训练，也没有产出权重。

## 1. 项目范围

```text
benchmark/   四套评测的数据、评测器、Harbor + Apptainer 运行环境和结果
sandbox/     ACP CPU Worker 上的沙盒控制系统，供 RL、评测等调用方远程执行
sft/         训练数据制作、训练入口和运行时校验
scripts/     数据下载、服务监控等辅助脚本
configs/     模型服务配置
outputs/     SFT 和 ACP 作业的提交、调度、取消记录
data/        原始训练数据快照（不进 git）
logs/        服务、下载和训练日志（不进 git）
```

## 2. 评测体系

### 2.1 覆盖的评测集

| 评测集 | 数据与规模 | 状态 | 当前结果 |
| --- | --- | --- | --- |
| Terminal-Bench 2.0 | 89 题，seed 42 | 暂停；最新一轮完成 79/89 | 通过 37、失败 23、超时 17、环境错误 2；剩 10 题未跑，最终通过率未产生 |
| HLE text-only | 全量 2158 题；固定 216 题子集 | 完成 | 全量答对 470 题，21.78%；216 题子集 43/216，19.91% |
| ASI-Bench seed42 | 60 题 × B1–B4，共 240 个条件 | 暂停 | 已跑 117 个条件：完成 97、超时 13、失败 7；正式分数需官方私有参考答案，尚未计算 |
| SWE-bench Pro V2 HARD-51 | 51 题 | 未开始 | 51 个镜像已准备并核验，尚未运行 |

TB2 本地使用 8 CPU / 32 GiB、单 seed；Qwen 官方模型卡的成绩是 32 CPU / 48 GB、5 次运行取平均。两者硬件和运行次数不同，不能直接比较。

### 2.2 公共基础工作

- 固定四套评测的数据和代码版本，记录在 `archive/benchmarks.lock.json`：TB2 代码 commit、89 题清单；HLE 代码 commit、HF revision、2500 题原始 Parquet；ASI 代码 commit、seed42 的 60 题；SWE 代码 commit、HF revision、642 题中的 HARD-51。
- 下载并校验数据：逐文件 SHA256、任务 ID、Parquet 行数和文件集合都写入 `archive/manifests/`，`scripts/verify_data.sh` 可重新计算并比对。HLE 原始 Parquet 与官方 revision 字节一致。
- 安装评测运行时：Harbor 0.24.0、Terminus-2 2.0.0、Apptainer 1.5.4。Apptainer 用官方 deb 解包到项目目录，不需要 sudo。
- 在官方版本上维护 9 个本地补丁，覆盖 Apptainer sessiondir、Singularity 启动等待、Apptainer 环境修复、控制进程独立解释器、公开源码缓存、响应观测、运行时提示符清理、Terminus 上下文恢复和 vLLM 前缀缓存隔离。补丁及安装后源码哈希记录在 `archive/harbor-local.lock.json`。
- 隔离和资源控制：TB2 每个 trial 使用独立网络命名空间和 slirp4netns，任务独占 localhost；评测、镜像准备和队列分别用 systemd 服务及 cgroup 限制 CPU、内存。
- 判分资源与 Agent 阶段分离：C4 分片、pyknotid/ocaml Git 缓存、cudnn wheel 等只在 Agent 结束后挂载，任务脚本和断言保持原样。

### 2.3 各评测集的镜像与环境

| 评测集 | 数据准备 | Docker 与镜像 | 运行方式 |
| --- | --- | --- | --- |
| Terminal-Bench 2.0 | 固定 Harbor 注册表的 89 题和任务清单 | 89 题引用 89 个不同官方镜像；按 linux/amd64 manifest digest 从镜像站拉取 OCI 层，用 Apptainer 离线转成 SIF，逐个校验结构和 SHA256；最后 3 个大镜像用 aria2 分段续传补齐 | Harbor + Terminus-2，Apptainer `singularity` 后端；每题 8 CPU / 32 GiB，3 小时，4 并发 |
| HLE | 274 MB 官方 Parquet 经字节校验后导出 2500 题全量 JSONL、2158 题 text-only JSONL，并抽出 1015 个图片文件；另按学科、题型和输入长度分层抽 216 题固定子集，用于训练过程中的快速迭代评测 | 不需要镜像 | 普通 HTTP 调用 4 台 vLLM 服务，每台 32 并发；生成后用官方 HLE prompt 和 `gpt-6.1-sol` 判分 |
| ASI-Bench seed42 | 固定官方代码和 seed42 的 60 题，B1–B4 共 240 个条件 | 60 题对应 41 个 `opencode_cli` 任务镜像，用官方 `TaskImageBuilder` 和任务 Dockerfile 在本地构建；Docker data-root 放在 32 GiB ext4 循环盘上，空间不足按 16 GiB 扩到最多 64 GiB | OpenCode 1.17.15，Docker `os` 沙盒，4 CPU / 32 GiB，8 并发，每条件 3 小时 |
| SWE-bench Pro V2 HARD-51 | 固定 51 题任务目录、补丁回放工具和官方测试 | 51 个官方公共镜像，逐个固定 manifest digest 后从 GHCR 镜像站拉取，校验 config ID、平台、源码仓库可达性和容器可启动；Docker data-root 在 32 GiB ext4 循环盘上，按需扩到最多 512 GiB | Agent 在宿主机，任务命令在 `--network none`、4 CPU / 32 GiB 的容器里执行；结束后在全新容器中回放补丁并运行原测试，两个阶段各 3000 秒 |

四套评测的镜像策略并不相同：TB2 是把官方预构建镜像转成 SIF，ASI 是在本地构建任务镜像，SWE 是按 digest 拉取官方镜像，HLE 完全不用镜像。共同点是镜像摘要、数据版本和任务清单都固定，并通过独立 Docker 存储和校验流程避免混用。

HLE 的 216 题子集是为训练迭代单独准备的快速评测集：

- 从 2158 题 text-only 全集中按 8 个学科 × 2 种题型分层，同时约束题目输入长度分布；长度排名 0–50%、50–90%、90–99%、99–100% 四段分别抽 108、87、19、2 题。
- 抽样只使用学科、题型、输入长度、ID 和 seed=42，不使用答案正确性或推理长度；题目内容与全集逐行一致，且全部无图。
- 固定任务 ID 和清单保存在 `archive/manifests/hle-text-only-10pct-seed42.json`，重复执行结果一致。
- 新 checkpoint 部署成独立的 vLLM 服务后，用 `run_hle.sh --model <模型名> --base-url <地址>` 跑这 216 题，结果按模型名写入 `results/<模型名>/hle-text-only-10pct/`；加 `--full` 才跑 2158 题，写入 `hle/`。两个目录独立，支持断点续跑，配置变化会拒绝混写。
- 同题基线为 43/216 = 19.91%，全量为 470/2158 = 21.78%。216 题里每多答对一题约增加 0.46 个百分点，适合训练中快速看趋势；小幅提升仍需全量确认，对外报告使用全量成绩。

### 2.4 运行与结果管理

- 提供镜像准备、SIF 修复、数据校验、后台启停、TB2 → ASI → SWE 接续队列和结果汇总脚本。
- 同一配置重跑时跳过已有终态结果；参数或代码变化时拒绝混写，并在结果目录保存配置、代码 SHA256 和 provenance。
- `archive/` 保留每轮修复和诊断的配置、日志、校验结果与历史记录；8.1 GB 的 `results/`、617 MB 的 `datasets/` 和第三方代码不进 git。

### 2.5 排查结论

vLLM 0.28 在 MTP、异步调度与前缀缓存同时开启时，命中缓存的请求可能读到错位的 GDN 状态，导致输出乱码。TB2 因此改为每次请求隔离缓存，并保留了两轮根因分析和 GPU 验证记录。

## 3. 沙盒控制系统

系统按调用方、控制器、宿主三层组织：

```text
调用方（RL rollout / 评测 / MiMo remote_apptainer / Harbor）
  │ 申请、续租、释放
  ▼
controller.py    选宿主、建删 ACP 宿主作业、管理租约、状态存盘
  ▼
host_server.py   每个 ACP CPU Worker 一个，承载多个沙盒
  └─ 每个沙盒：私有可写 rootfs + apptainer exec 里的 process_agent.py
```

已完成的能力：

- 按镜像分配沙盒，优先选择已经解包该镜像的宿主；宿主不够时自动提交新的 ACP 纯 CPU 作业，空闲宿主自动删除。
- 租约创建、续期、释放和到期回收；控制器重启后接管仍存活的宿主和租约；孤儿沙盒清理。
- 沙盒内支持普通命令、常驻 Python/Bash 会话、PTY 终端、分块文件读写和整体重置。
- 调用方客户端和命令行 `client.py`，以及镜像缓存 `images.py`、控制器入口 `run_controller.sh`。
- 检查脚本：`check_pool.py` 覆盖并发申请、隔离、token 权限、续期、释放和租约到期；`check_control.py` 覆盖单个沙盒的 17 项控制行为。

验证情况（2026-10-20，记录在 `claude_workspace/2026-10-20_沙盒控制系统/`）：

- ACP 上控制器从 0 台扩到 2 台宿主，一次申请 5 个沙盒，34 秒内全部建好；宿主就绪分别用 11 秒和 23 秒。
- 首次使用某镜像建沙盒约 6–9 秒，复用后约 2.5 秒。
- `check_pool.py` 7 项、`check_control.py` 17 项和 MiMo `remote_apptainer` 转发 6 项通过。
- kill -9 控制器后重启能接管宿主和租约；空闲宿主被删除；SIGTERM 控制器时宿主作业被删除。

已知限制：

- `/sys/fs/cgroup` 只读，单沙盒无法限制 CPU 和内存，失控任务会影响同宿主其他沙盒。
- 默认配置（32 vCPU / 128 GiB、每台 8 槽位）和 64 路并发尚未实测。
- 控制器部署在 ACP 上尚未实测；`images.py pull` 尚未完整拉取过镜像。
- 沙盒内没有 `/proc`，对依赖它的任务影响尚未系统验证。

## 4. SFT 数据与训练

已完成：

- 下载并校验 9 个上游数据仓库，原始快照约 452 GiB，记录固定 commit、文件清单和内容哈希。
- 制作 12 个训练 JSONL，共约 121 GB，覆盖 Math、Science、Proofs、PDBThink、MindForge、RTS、RLMath、NatureBench 等来源，并完成格式、行数、SHA256 和来源校验。
- 提供两套训练入口：Science + Math 使用 2 节点 × 8 卡、64K、packing、global batch 32；Agent 分 positive / full 两版，使用 16 卡、256K、packing、TP/SP 16、EP 16、micro batch 1 / global batch 8。
- 运行时包含缓存准备、磁盘保护、每 100 步完整 checkpoint 和最终 HF 权重校验。

实际状态：

- r01（Science + Math）、r02（Agent positive/full）、r03（重提）都已提交到 ACP。
- 截至 2026-10-17 的记录，任务因 GPU 配额不足停留在 STARTING；r02 已取消，r03 保留但 `actual_training_started` 为 false。
- `ckpts/` 为空，尚未产出任何训练权重。

## 5. 模型服务

- 使用 Qwen3.6-35B-A3B，4 张 A800-80GB，TP=4，端口 8200，262144 上下文，启用 MTP、`qwen3` reasoning parser 和 `qwen3_xml` tool parser。
- 曾部署主服务和 3 个四卡副本，HLE 全量评测使用 4 台服务、每台 32 并发，共 128 并发。
- 主服务已于 2026-10-19 按用户要求停止，作业已删除。仓库保留启动脚本 `run_vllm_serve.sh`、配置 `configs/` 和服务日志位置，不包含模型权重。

## 6. 仓库内容与数据边界

| 内容 | 是否进 git | 说明 |
| --- | --- | --- |
| 代码、脚本、配置、README | 是 | 评测、沙盒、SFT、服务入口 |
| `benchmark/archive/` | 是 | 约 8 MB，版本锁、补丁、验证证据和历史记录 |
| `outputs/` | 是 | ACP 作业提交、调度和取消记录 |
| `sft/data_samples/` | 是 | 12 个训练文件各前 10 行，共约 28 MB |
| `sft/data/` | 否 | 完整训练 JSONL，约 121 GB |
| `data/training/` | 否 | 原始数据快照，约 452 GiB |
| `benchmark/datasets/` | 否 | 617 MB 固定版本评测数据 |
| `benchmark/results/` | 否 | 8.1 GB 评测结果 |
| 模型权重、虚拟环境、缓存、日志 | 否 | 由 `.gitignore` 排除 |

## 7. 未完成事项

- TB2 剩余 10 题未跑，现有 37/89 不是最终成绩。
- ASI-Bench 还剩 123 个条件未跑，正式分数需要官方平台计算。
- SWE-bench Pro 尚未开跑。
- SFT 训练未真正开始，没有权重；需要等 GPU 配额。
- 沙盒的单沙盒资源限制、64 路并发、控制器在 ACP 上部署、镜像站完整拉取和 `/proc` 兼容性都未完成。
- 模型服务已停止，重新评测前需要重新部署。

## 8. 复现入口

各目录的 README 记录了完整命令和当前配置：

- `benchmark/README.md`：四套评测的运行、镜像准备、结果索引和已知问题。
- `sandbox/README.md`：控制器启动、客户端用法、HTTP API、生命周期和 ACP 限制。
- `sft/README.md`：数据制作、校验、两套训练入口和输出位置。

常用检查：

```bash
cd benchmark
bash scripts/verify_data.sh
.venv-harbor/bin/python -B tests/test_eval_hle.py
.venv-harbor/bin/python -B tests/test_eval_asi.py

cd ../sft
/mnt/data/rui.yan/venvs/OmniRL/bin/python prepare_data.py --check
```

沙盒检查需要先按 `sandbox/README.md` 启动控制器，再运行 `check_pool.py` 或 `check_control.py`。
