# Benchmark 评测历史记录

本文件是 2026-10-20 整理前 `benchmark/README.md` 的原文，按时间顺序记录各轮评测、修复与诊断，内容未删改。当前用法和状态见 `../README.md`。

文中路径保持当时的写法，与现在布局的对应关系：

| 当时 | 现在 |
| --- | --- |
| `archive/scripts/manage_tb2.sh`、`manage_asi.sh`、`verify_data.sh` 等维护脚本 | `scripts/` |
| `archive/scripts/test_eval_*.py`、`archive/tests/` | `tests/` |
| 顶层 `run_tb2_*_retest.sh`，`scripts/run_tb2_fixed89.py`、`probe_tb2_prefix_cache.py`、`capture_tb2_model_responses.py`、`preflight_tb2_environment.py`、`tb2_environment_status.py` | `archive/tb2-diagnostics/` |
| `coding_agent/data/benchmark/` | `coding_agent/benchmark/` |

---


四套评测的数据、官方评测器、官方运行脚本、本地运行工具、依赖环境、镜像管理软件、配置和日志统一放在本目录。这里是现有 `coding_agent` 仓库中的一个目录。

```text
benchmark/
├── run_tb2.sh                 TB2.0 评测入口，显式列出核心参数
├── run_hle.sh                 HLE 生成、判分与汇总入口
├── run_asi.sh                 ASI 评测入口，显式列出核心参数
├── run_swe.sh                 SWE HARD-51 官方镜像及独立 Docker 管理
├── datasets/                 四套固定版本数据
├── third_party/              官方评测代码及运行脚本
├── scripts/                  运行环境设置及各评测集启动代码
├── results/<模型名>/<评测集>/  逐题结果、轨迹与评分
├── archive/                  版本记录、协议、校验凭据与维护脚本
├── .venv-harbor/              已安装的 Harbor 与 Terminus-2
├── .sandbox-tools/apptainer/  已安装的 Apptainer、辅助程序和动态库
├── .cache/docker-images/asi-bench/  ASI Docker 原生镜像存储
└── .data-tools/               已安装的数据处理依赖
```

## 当前环境

Harbor `0.24.0`、Terminus-2 `2.0.0`、Apptainer `1.5.4` 已安装，Harbor 选择 `singularity` 后端。框架和 Agent commit 为 `b53b8134e1241686dca7759af188f987ecc48e8b`；Harbor 依赖锁见 `archive/requirements-harbor.lock.txt`，数据处理依赖 `pyarrow==21.0.0` 单独安装于 `.data-tools/`。软件版本、安装凭据和最近自检结果统一记录在 `archive/harbor-local.lock.json`。Harbor 后端另有本地补丁 `archive/patches/harbor-0.24.0-singularity-startup.patch`，使首次安装依赖的启动等待时间可配置；默认仍为上游的 60 秒，本地冒烟检查设为 300 秒。Terminus-2 已回补上下文恢复修复 `archive/patches/harbor-0.24.0-terminus-context-recovery.patch`，来源为上游 PR #3310，并适配 80K 输出预留；它是基于官方版本的本地修复构建，补丁及安装后源码哈希均已记录。数据来源和固定版本记录在 `archive/benchmarks.lock.json`，任务 ID 和逐文件哈希在 `archive/manifests/`。归档内容供版本复核、数据校验与环境重建使用。

`archive/scripts/manage_tb2.sh harbor` 通用检查入口使用本机 systemd/cgroup，单并发时把 Harbor 和任务子进程合计限制在 **4 CPU 配额、8 GiB 内存**。`archive/harbor-local.json` 保留单并发、300 秒 Agent 超时的 Oracle 自检示例，需通过 `-c` 显式加载；资源由外层 cgroup 落实，因此示例中的 Harbor 资源策略为 `ignore`。TB2.0 评测队列的每任务资源另由 `run_tb2.sh` 开头的参数指定，当前为 4 CPU / 32 GiB。数据校验入口单独限制为 2 CPU、4 GiB。HLE 入口直接在本机运行 Python，通过 HTTP 请求模型服务，无需沙盒，不额外设置 CPU 或内存限制。

本目录是 Linux/Python 3.12 本地环境；Harbor 和数据校验入口需要本机 systemd/cgroup 权限，HLE 入口无需这些权限。已有 vLLM 服务通过 HTTP 调用，接口信息保存在 `archive/model-endpoint.json`，调用时显式传入服务地址和模型名。模型权重和服务由独立的模型部署流程管理。

## 使用

在本目录执行以下命令；运行参数中的相对路径以本目录为基准。

```bash
cd /mnt/data/rui.yan/projects/coding_agent/benchmark
bash archive/scripts/manage_tb2.sh harbor --version
source scripts/harbor_env.sh
singularity --version
```

只解析一个任务的配置，不启动任务或下载任务镜像：

```bash
bash archive/scripts/manage_tb2.sh harbor run -c archive/harbor-local.json \
  -p datasets/terminal-bench-2.0/fix-git --print-config
```

需要维护数据时，使用归档中的校验入口：

```bash
bash archive/scripts/verify_data.sh
```

固定的 Harbor `0.24.0` 通过 `--ak api_base=http://10.119.27.14:8200/v1` 为 Terminus-2 指定模型服务地址。`scripts/harbor_env.sh` 会清除代理设置；所有下载必须直连。维护脚本集中在 `archive/scripts/`：HLE 转换、数据校验及 Apptainer 安装器。安装器将官方发布包及 Ubuntu 辅助依赖解包到本目录，并记录校验值与本地路径配置，不安装系统级软件包。

## 数据与评测协议

| Benchmark | 数据路径（相对 `datasets/`） | 固定范围 |
|---|---|---|
| SWE-bench Pro V2 HARD | `swe-bench-pro-v2-hard51/` | 51 个独立任务目录与 HARD Parquet |
| HLE | `hle/test.jsonl`、`hle/text_only.jsonl`；原始文件 `hle/data/test-00000-of-00001.parquet` | 完整 test 2500 题；text-only 2158 题 |
| ASI-Bench | `asi-bench-seed42/` | seed42，60 题、B1–B4 共 240 个条件；参考答案私有 |
| Terminal-Bench | `terminal-bench-2.0/` | 固定 Harbor 注册表 `terminal-bench@2.0`，89 题 |

HLE 原始 Parquet 与固定 HF revision 字节一致；JSONL 保留所有文字字段与原始顺序。text-only 按原始 `image` 字段为空筛选，移除 `image_preview`、`rationale_image`；其中 254 题仅参考解析有图，文字解析仍保留。完整集图片原样放在 `hle/images/`，JSONL 使用相对路径。评测只将题目输入模型，答案和解析用于评分。text-only 成绩单独报告。

HLE 重导出：

```bash
PYTHONPATH="$PWD/.data-tools" .venv-harbor/bin/python archive/scripts/export_hle_jsonl.py
```

SWE V2 正式成绩使用离线 Agent 阶段捕获的补丁，在全新沙盒中重评分。ASI seed42 官方评分要求 Docker `--sandbox os` 的可验证运行记录，并在官方平台使用私有参考答案评分；当前 Apptainer 环境不替代这项要求。Terminal 完整评测应覆盖固定任务集，并记录种子、资源和模型配置。小资源环境检查配置不作为完整评测配置。

镜像按各评测集的固定任务清单准备，SWE HARD-51 与 ASI 分别使用独立 Docker 服务。Terminal-Bench 固定的 89 个任务引用 89 个不同镜像，现已全部准备并校验；83 题历史评测及其后的 9 题诊断复测均已完成，当前正在补齐全部 89 题，详见下文全量评测说明。HLE text-only 结果保存在 `results/<模型名>/hle/`。Harbor 结果位于 `results/<模型名>/terminal-bench-2.0/`，`.cache/harbor-images/` 保存可复用的 SIF，`.cache/harbor-assets/` 保存经校验的固定版本依赖；辅助配置与临时文件统一进入 `.cache/tmp/`。Apptainer 的重复 OCI 下载缓存默认关闭。源数据、第三方代码、缓存和依赖环境不进入 Git；配置、清单与本地脚本可跟踪提交。

入口默认禁用 pip 缓存，运行任务所需的镜像缓存按需生成。

## Terminal-Bench 2.0 镜像复用

`.cache/harbor-images/` 中的 SIF 是各任务的完整只读环境。每次运行使用独立的临时写入层，结束后丢弃；后续种子和同环境模型可以复用同一个 SIF。旧冒烟检查及暂停运行的结果已按用户要求清空，当前结果目录仅保存本轮运行。

DNS、证书和软件源配置只读挂载，公共运行依赖保存在 `.cache/harbor-assets/`，不会随结果清理。官方任务和测试脚本保持原样。

## ASI-Bench 评测

`run_asi.sh` 与 TB2 入口保持相同结构：环境设置加一条 Python 调用。
模型名、服务地址列表及凭据环境变量在 `--agent-config` 中；数据、输出目录、
B1–B4、seed、并发、超时、工具模式和资源限制均直接列在脚本里。
模型通过本机 vLLM 的 OpenAI-compatible 接口调用，Agent 固定为 OpenCode `1.17.15`，
默认 seed42 的 60 题 × B1–B4 共 240 个条件，最多 8 个沙盒并发、每任务 3 小时、4 CPU / 32 GiB。
`api_base` 接受单个 URL 或 URL 数组，目前默认只使用与 TB2 相同的主服务 `http://10.119.27.14:8200/v1`。
每个任务条件启动时选择当前负载最少的服务，整次尝试固定使用该服务；
默认 `--parallel 8`，可调小至 1，最大值为 8；配置多个服务时按负载分配条件，结束后释放槽位。
每次尝试单独创建官方 OpenCode 适配器，线程之间隔离服务地址及 Docker 镜像凭据。
运行时 ASI 专用 cgroup 的总上限为 32 CPU / 256 GiB，内存按需使用；
`--parallel` 或每沙盒资源变化时，总上限按二者乘积设置。
`--timeout` 是每个任务条件的总时间预算；模型采样及轮数使用固定版本 OpenCode 的默认行为，
当前入口没有额外覆盖上下文、输出 token 或采样参数。

```bash
bash run_asi.sh --print-config  # 预览参数，不启动 Docker 或调用模型
bash run_asi.sh --check         # 检查数据、固定版本、镜像 ID 和模型服务，不执行任务
bash run_asi.sh                 # 正式生成任务产物及轨迹

# 单题 B1；另设输出目录，避免与完整集合混用
bash run_asi.sh --tasks math.sigma_linear_diophantine --prompt-levels b1 \
  --output-dir results/Qwen3.6-35B-A3B/asi-bench-smoke

# 保持原配置，续跑尚无结果的条件
bash run_asi.sh --resume results/Qwen3.6-35B-A3B/asi-bench
```

相对路径以本目录为准。`--instances-dir` 必须指向包含 `任务ID__seed42/` 的
`datasets/asi-bench-seed42/tasks/`。后追加的官方参数覆盖前面的同名参数；
覆盖 Agent 配置时传入完整的 `--agent-config` JSON，凭据只填写环境变量名。
例如：`"api_base": ["http://服务器1:8200/v1", "http://服务器2:8200/v1"]`，
同一服务池使用相同的模型名、协议及凭据环境变量；`--check` 会检查池内每个服务。
无鉴权的本地 vLLM 默认使用 `ASI_MODEL_API_KEY=EMPTY`，也可在本机设置该变量。
`--cpus`、`--memory-mib`、`--print-config` 和 `--check` 是本地启动器参数，
其余参数沿用官方 `asibench run`，帮助见 `bash run_asi.sh --help`。

启动器保留配置、源码版本、镜像检查和状态于输出目录的 `launcher.json`，
官方 runner 保存逐条件结果、产物、轨迹、模型调用记录和运行元数据。
`endpoint_assignments.jsonl` 记录每次尝试使用的服务器、开始/结束事件和状态；
官方逐次模型调用记录也保留实际服务地址。服务分配适配器的源码哈希写入 `launcher.json`。
已有结果必须显式 `--resume` 且保持配置一致；配置变化应另设输出目录。
官方续跑会跳过已有结果，包括已记录的失败和超时；重试失败条件需使用独立输出目录。
seed42 的参考答案私有，任务运行成功只代表产物已收集，正式分数由 ASI 官方平台计算。
可以查看运行报告，或只在本地生成提交包：

```bash
bash archive/scripts/manage_asi.sh cli report \
  --results-dir results/Qwen3.6-35B-A3B/asi-bench

bash archive/scripts/manage_asi.sh cli submit \
  --results-dir results/Qwen3.6-35B-A3B/asi-bench \
  --benchmark-repo Apexintelligence-AI/ASI-Bench-seed42 --no-upload
```

### ASI 镜像与 Docker 管理

ASI 使用独立 Docker 服务，**Docker 原生 data-root 就是
`.cache/docker-images/asi-bench/`，不导出镜像包**。系统 Docker 保持独立。
`scripts/asi_env.sh` 设置专用 socket；维护入口 `archive/scripts/manage_asi.sh`
在需要时启动独立 Docker 和 containerd，评测入口在运行前调用同一环境检查。
共享挂载盘不适合直接写入 Docker 的大量小文件：100 个文件的实测写入耗时
约 3.55 秒，ext4 挂载下约 0.03 秒。因此将同目录旁的 `asi-bench.ext4` 底层磁盘
挂载到原生 data-root，Docker 使用标准 `overlay2`。初始容量 32 GiB，构建时
空间不足会按 16 GiB 扩容，最多 64 GiB；该文件是 Docker 的底层存储，不是导出包。
入口在重启后自动恢复挂载，再启动独立 Docker 和 containerd。

```bash
bash archive/scripts/manage_asi.sh docker info --format '{{.DockerRootDir}}'
bash archive/scripts/manage_asi.sh docker images
bash archive/scripts/manage_asi.sh prepare-images
bash archive/scripts/manage_asi.sh cli --help
```

seed42 的 60 个任务对应 41 个 `opencode_cli` 任务镜像，Agent 固定
`opencode 1.17.15`。B1–B4 复用任务镜像，每次运行使用独立容器及工作目录。
准备脚本只构建和检查镜像，不运行模型评测；已有本地镜像直接复用。
Dockerfile、镜像 ID、任务映射、软件版本和日志保存在 `.cache/asi-build/`，
后台服务为 `coding-agent-asi-images.service`，总日志是 `.cache/asi-build/prepare.log`。

仅构建镜像时 Docker 和 containerd 默认共用 4 CPU / 32 GiB 配额，构建单并发，每步同样限制
4 CPU / 32 GiB。`run_asi.sh` 将 ASI 任务沙盒默认设置为 4 CPU / 32 GiB，
通过本地入口覆盖默认值，保留固定版本的上游源码。独立 Docker 不管理宿主 bridge 或 iptables，构建及 Agent API
访问使用 host 网络，工具模式由 ASI 配置控制。下载清除六项代理变量并禁用代理
build args；Debian 和 PyPI 使用直连 Aliyun 镜像，保留 Debian 签名校验。
uv 安装站直连返回 403，因此使用 PyPI 的 `uv==0.8.22`；任务包版本要求保持不变。
Dockerfile 及已安装版本记录可用于重建，三组模型对比时复用相同镜像 ID。

## SWE-bench Pro V2 HARD-51 镜像准备

`run_swe.sh` 管理独立 Docker / containerd，原生 data-root 为
`.cache/docker-images/swe-bench-pro-v2-hard/`，与 ASI 和系统 Docker 分开。
固定 HARD-51 的 51 个任务分别引用官方公共镜像
`ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`。准备脚本先固定 linux/amd64
manifest digest，再通过直连 GHCR 镜像源 `ghcr.nju.edu.cn` 按该 digest
拉取并保留原始任务标签；配置 ID 及各文件层由 Docker 校验，重复执行复用已有镜像与固定 digest。
镜像源连续失败两次后回退到官方 `ghcr.io`；此前已完成的镜像直接复用，每个镜像的实际下载来源保存在 `download_ref`。
每个镜像检查配置 ID、平台、源码仓库能否访问，以及 4 CPU / 32 GiB 容器限制。

```bash
bash run_swe.sh prepare-images --background
bash run_swe.sh status
bash run_swe.sh docker images
```

后台服务为 `coding-agent-swe-images.service`，下载与 Docker / containerd 共用
4 CPU / 32 GiB 总配额，最多两个任务重叠准备，Docker 层下载并发上限为四个。
控制器单独串行写入清单，磁盘扩容使用线程锁。网络直连，清除
六项代理变量并单独配置不含凭据和代理的 Docker client。任务及上游源码保持原样。
任务映射、固定 digest、原始 manifest、镜像 ID 和检查结果保存在
`.cache/swe-build/manifest.json` 及 `metadata/`，逐任务日志在同目录的 `logs/`。
全部镜像就绪后自动核验 51 个官方标签、固定 digest、任务文件及容器检查凭据，结果写入 `metadata/job.json` 的 `completion_verification`。
脚本只准备镜像，不运行模型评测。

与 ASI 一样，原生目录挂载于旁边的 `swe-bench-pro-v2-hard.ext4` 底层磁盘文件，
使用标准 `overlay2`，不导出镜像包。磁盘初始容量 32 GiB，按需要每次扩容
32 GiB，磁盘容量上限 512 GiB；稀疏文件按实际写入占用空间，和内存配额无关。
`run_swe.sh run` 现已接入离线 Agent 阶段和全新沙盒重评分。SWE 采用固定版本的上游 `host_mini_swe.py` 和 `patch_replay.py`，mini-swe-agent `2.4.6` 独立安装在 `.cache/swe-agent-runtime/`，不修改 TB2 的依赖环境。LLM 循环运行在宿主，任务命令仅在 `--network none`、4 CPU / 32 GiB 的 Docker 容器内执行，模型凭据不传入容器。每题 Agent 预算 3000 秒；超时也捕获已有 `git diff`，销毁 Agent 容器后，在同一固定镜像的新容器中通过官方 `PatchReplayAgent` 回放补丁，再运行原始 `tests/test.sh`（判分预算 3000 秒）。只在判分容器中放入官方测试，产物中记录两只容器的不同 ID、镜像 ID、补丁和判分日志。

默认模型为主服务上的 `Qwen3.6-35B-A3B`，采用上游 textbased mini 配置、seed42、temperature=1.0、top_p=0.95、top_k=20、thinking=on 和 81920 输出上限，按 8 题滚动并发。不设自定义步数或货币成本上限；宿主进程和命令超时共同落实每题 50 分钟预算。结果保存到 `results/Qwen3.6-35B-A3B/swe-bench-pro-v2-hard51/`，`progress.json` 逐题记录，`summary.json` 在 51 题全部结束前不填写最终通过率。相同配置重启时跳过已有终态结果，配置不一致则拒绝混写。

```bash
bash run_swe.sh check  # 核对 51 题、固定镜像和运行依赖
bash run_swe.sh smoke  # 无模型调用的离线隔离、补丁捕获与新容器回放检查
bash run_swe.sh run    # 正式运行 HARD-51 并判分
```

无模型调用的真实 Docker 检查凭据为 `archive/validation/swe-locked-docker-smoke.json`；交接条件、ASI 全集覆盖和判分异常检查位于 `archive/tests/test_evaluation_queue.py`。

## TB2 → ASI → SWE 后台接续

2026-10-17 按用户要求建立 `coding-agent-eval-queue.service`，顺序为当前 TB2 全量补齐、ASI seed42 的 60 题 × B1–B4 全部 240 个条件、SWE Pro V2 HARD-51 全部 51 题，后两套评测均使用主服务。交接必须同时满足 TB2 的 `complete=true`、已结束数 89、任务数 89，以及 `coding-agent-tb2.service` 已退出，83 题历史结果不会触发交接。准备阶段的 Docker 检查不调用模型。

队列入口是 `scripts/run_eval_queue.py`；计划、运行状态和日志保存在 `results/Qwen3.6-35B-A3B/evaluation-queue/`：`plan.json` 保存固定范围与检查凭据，`progress.json` 记录当前阶段，`asi.log`、`asi-package.log`、`swe.log` 保留执行日志。ASI 首次运行调用 `run_asi.sh`，已有结果则显式 `--resume`；检查 240 个条件的最后一次尝试，不按最好分挑选结果。全部条件尝试完后生成本地官方提交包（`--no-upload`），seed42 的官方私有参考答案分数保持待判状态，然后继续 SWE。SWE 的报告使用全新沙盒中原始官方判分器的结果。

```bash
systemctl status coding-agent-eval-queue.service --no-pager
.venv-harbor/bin/python scripts/run_eval_queue.py --status
journalctl -u coding-agent-eval-queue.service -n 30 --no-pager
```

## Terminal-Bench 2.0 全量补齐：89 个任务、seed42

2026-10-17 按用户要求补齐全部 89 题，并重跑原评测中所有出现上下文压缩失败的任务。当前 `run_tb2.sh` 默认输出到 `results/Qwen3.6-35B-A3B/terminal-bench-2.0/full-seed42-context-fixed/`，其中 `task-ids.txt` 与固定的官方 89 题清单一致。仅使用主服务 `http://10.119.27.14:8200/v1`，8 个任务滚动并发，每题 4 CPU / 32 GiB、3 小时，采样、thinking 和 token 预算沿用原评测。

这次需要新运行 **16 题**：补齐此前未评测的 6 题 `mteb-leaderboard`、`mteb-retrieve`、`qemu-alpine-ssh`、`reshard-c4-data`、`train-fasttext`、`write-compressor`，另重跑 **10 个**原日志出现 `Full summary failed` 或 `Even fallback chat failed` 的任务。筛选覆盖原 83 题的全部最终状态；除此前 9 个超时任务外，还包括报执行错误的 `feal-linear-cryptanalysis`。此前 9 题诊断复测的结果不参与本轮选择，10 题均重新执行。

其余 **73 题**复用原始结果，包含原来的通过、失败、超时和执行错误；任务目录通过符号链接保留原始轨迹。全量结果的 `provenance/full-coverage-plan.json` 保存 83 题逐日志校验、筛选依据、逐题来源、旧结果与补丁哈希；`provenance/prelaunch-verification.json` 记录 16 个镜像的 SHA256 校验和汇总检查。73 题保留原 Agent 版本，16 题使用 `context-recovery.1`，汇总会披露这一组成。

后台服务仍为 `coding-agent-tb2.service`，通过 `bash archive/scripts/manage_tb2.sh status` 查看状态，通过 `bash archive/scripts/manage_tb2.sh summary` 查看当前 89 题汇总。全量目录的 `summary.json` 和 `seed42/progress.json` 持续更新；全部结束后生成 `seed42/report.md` 和 `seed42/tasks.csv`。最终分母固定为 **89**，包含错误与超时，未完成前不填写最终通过率。这是单 seed、修改资源条件下的全任务覆盖评测，保留与官方硬件和多次运行口径的差异。

## Terminal-Bench 2.0 历史评测：83 个任务、单种子

评测入口 `run_tb2.sh` 仅保留运行环境和一条 Python 调用，模型名、服务地址、数据和结果路径、种子、并发、每任务资源、超时、步数及采样预算均显式列出；可直接编辑，也可追加同名命令行参数覆盖。`scripts/harbor_env.sh` 仅设置沙盒路径、依赖运行环境和直连网络。后台启停、镜像准备与 Harbor 检查集中在 `archive/scripts/manage_tb2.sh`，原复杂入口保存在 `archive/run_tb2-managed.sh`。

只查看最终配置，不启动评测：

```bash
bash run_tb2.sh --print-config
```

**旧结果已清空，本轮使用启动时已就绪的 83 个镜像运行 seed 42，已于 2026-10-17 11:40（北京时间）完成。** 结果为 31 个通过、25 个判分失败、19 个超时、8 个执行错误，通过率 **31/83 = 37.35%**。任务清单固定在 `results/Qwen3.6-35B-A3B/terminal-bench-2.0/task-ids.txt`，选择凭据保存在同目录 `provenance/selection.json`。这是 83/89 的子集，汇总明确保存 `is_full_task_set=false`；分母包含全部 83 个已选任务（含失败和超时），不作为完整 TB2.0 成绩。后续补齐的镜像不自动改变本轮清单。

19 个超时的逐项分析见同一结果目录的 `provenance/timeout-analysis.json`：18 个 Agent 3 小时超时、1 个判分超时。9 个任务出现反复上下文超限和摘要恢复失败，增加沙盒 CPU/RAM 无法直接解决；另有长生成/输出截断、终端交互、外部网络与端口隔离问题。`query-optimize` 的底层 HTTP 默认 600 秒超时先于官方 900 秒验证预算触发。单独归为执行错误的 `count-dataset-tokens` 有 32 GiB cgroup OOM 记录。原始结果不改写；日志及内存证据保存在 `provenance/`。

2026-10-17 核对的 Qwen 官方模型卡报告 TB2.0 为 **51.5%**，并注明 Harbor/Terminus-2、32 CPU / 48 GB、3 小时、80K 输出、256K 上下文及 5 次运行平均。官方模型卡经直连 HF 镜像取得，快照和下载凭据保存在 `.cache/harbor-images/records/terminal-bench-2.0/qwen36-official-model-card-20261017.*`；本轮子集、硬件和运行次数与该口径不同。

```bash
bash archive/scripts/manage_tb2.sh images start
bash archive/scripts/manage_tb2.sh images status
# 停止镜像准备：
bash archive/scripts/manage_tb2.sh images stop
```

镜像准备由 `coding-agent-tb2-images.service` 后台运行，默认最多 **8 路并行**，整体限制 **16 CPU / 96 GiB**。解包暂存使用 `/dev/shm`，并按镜像大小预估限制并行暂存总量为 80 GiB，避免共享文件系统上大量小文件写入；临时数据完成后清理，内存按需占用。成品始终保存在 **`.cache/harbor-images/`**，此构建流程保留全部成品，不按旧的 20 GiB 试跑限制淘汰。

进度、逐镜像日志、OCI digest、SIF SHA256 及版本凭据保存在 **`.cache/harbor-images/records/terminal-bench-2.0/`**；`progress.json` 中的 `complete=true` 表示全部 89 个镜像已准备并校验。镜像准备不会调用模型或运行评测。下载默认使用经相同 manifest digest 核验的 `docker.1panel.live`，重试交替使用 `docker.1ms.run`；强制 HTTP/1.1，并在 300 秒无下载或输出进展时重试，避免 HTTP/2 流错误后长时间挂住。镜像内容始终按已记录的 OCI digest 固定。网络传输均清除代理设置，运行时入口直接使用 Apptainer，没有串行构建锁。

当前运行结果保存在 `results/Qwen3.6-35B-A3B/terminal-bench-2.0/seed42/`，每个任务记录固定的服务地址、原生 Harbor 配置、轨迹、评分及 token 用量。已完成任务在同协议下断点续跑时不重复执行。

结果根目录直接放 `seed42/`、`seed43/` 等 seed 目录；当前只启用 seed42。根目录 `summary.json` 汇总每个 seed 的成功率及最终均值、标准差，`completed_seeds` 表示已完成的 seed 数。全部已选 seed 完成前不填写最终均值；单 seed 完成后的均值就是该 seed 的成功率，不估计跨 seed 标准差。

每个有对话的任务可直接打开 `seed42/<任务名>/trajectory.json`，其中 `steps[].message` 是消息正文，`reasoning_content` 是模型思考，`tool_calls` 和 `observation` 是命令与终端返回；`result.json` 是该任务的原生判定结果。这两个文件是指向原始输出的符号链接，不复制对话或评分数据。尚未进入 Agent 阶段的任务可能没有对话文件。原生 `attempt*/.../agent/` 与 `verifier/` 仍保留完整审计记录。

本轮已按用户要求清空旧结果并从头重跑，直接写入 `seed42/`。评测协调器更新根目录 `summary.json` 和每任务的对话、判定快捷链接。启动时间、固定的 83 个任务清单及清空重跑凭据保存在 `provenance/selection.json`。


本轮只启用 **seed 42、83 个任务**。每任务 **4 CPU / 32 GiB**、**总计 8 个任务滚动并发**、Agent **3 小时**、无自设步数上限，判分采用官方任务超时。temperature=1.0、top_p=0.95、top_k=20、thinking 默认开启，单次输出 **81920 tokens**，context **262144 tokens**。当前硬件、单种子和任务子集口径与预定完整评测配置不同。

已完成的 83 题评测使用四台同模型服务。2026-10-17 按用户要求将 `run_tb2.sh` 默认地址改为主服务 `http://10.119.27.14:8200/v1`，总并发仍为 8。`--base-url` 仍可列出多个同模型服务；任务启动时选择当前任务数最少的一台，整个任务的所有 Agent 请求固定使用该地址。服务端需启用 prefix caching，实际命中也受缓存容量和淘汰影响。

`bash run_tb2.sh` 在前台运行；后台运行使用 `bash archive/scripts/manage_tb2.sh start`，协调器限制为 2 CPU / 4 GiB，各任务独立执行上述资源限制。维护入口的 `stop`、`status` 和 `summary` 分别用于停止、查看服务状态和汇总已有结果。

## Terminal-Bench 2.0 大镜像下载修复

最后三个大镜像已由 `archive/scripts/repair_tb2_images.py` 补全，2026-10-17 12:02（北京时间）达到 **89/89 全部就绪**，均已校验。维护入口为：

```bash
bash archive/scripts/manage_tb2.sh images repair
bash archive/scripts/manage_tb2.sh images status
```

后台服务 `coding-agent-tb2-image-repair.service` 限制为 8 CPU / 64 GiB；本地转换暂存预算 48 GiB，按镜像大小控制同时转换数量。现有评测继续使用固定任务清单。已完成的 SIF 不覆盖。

下载先固定原任务的 linux/amd64 manifest、config 和全部层 SHA256。`qemu-alpine-ssh` 已从 `dockerproxy.net` 下载完成并通过沙盒启动检查；该源的两个 MTEB 大层在约 1 GiB 处中断，且忽略 Range。修复脚本改用经非零偏移和文件末尾 Range 验证的 `docker.amingg.com`，由 aria2 每层最多 4 条连接续传，两个大层共最多 8 条连接；不沿用一小时整体下载上限。已有连续前缀转交 aria2 继续使用，完成片段由控制文件持久保存，重试时刷新公共镜像的匿名拉取令牌。所有下载清除代理、验证 TLS 和最终 SHA256。

完整层到齐后生成本地 OCI layout，再离线转换为 SIF，使用 zstd level 3 压缩；检查 SIF 结构、沙盒能否启动及最终文件 SHA256，全部通过才标记就绪。三个镜像补齐后清理重复的临时 OCI 层，保留最终 SIF 和校验凭据。

当前逐层字节进度在 `.cache/harbor-images/records/terminal-bench-2.0/resumable-download-progress.json`；日志是同目录 `resumable-repair.log`，分段下载日志是 `<任务名>.range-download.log`，本地构建日志是 `<任务名>.repair-build.log`。失败和重试历史保存在 `resumable-repair.json`。

## Terminal-Bench 2.0 上下文恢复补丁

本机 Harbor 0.24.0 的 Terminus-2 已回补 [上游 PR #3310](https://github.com/harbor-framework/harbor/pull/3310)，适配当前 options 接口和原生输出截断处理；摘要请求和完整交接内容同时计算提示词与显式输出预留，预算修正参考 [Marin PR #55](https://github.com/marin-community/harbor/pull/55)。修复不会改变 256K 上下文、80K 输出上限或 thinking 设置。当前 Agent 属于有明确补丁记录的本地构建，不宣称源码与官方版本完全一致。

备用摘要会替换旧历史，保留初始指令、原始任务及终端状态；恢复请求仍超限时向上报告错误。主动摘要会为输出预留空间，传输重试保留实际交接提示，输出截断继续使用 0.24.0 原有处理方式。沙盒中的文件保持原状。

补丁为 `archive/patches/harbor-0.24.0-terminus-context-recovery.patch`。旧版本关键回归测试为 3 失败 / 1 通过，修复后 23 项通过；真实 Qwen 服务上的正常三阶段摘要和强制短摘要备用路径均通过，测试使用 thinking=on、262144 context、81920 输出上限，不执行任务命令。源码哈希、来源和测试凭据在 `archive/validation/terminus-context-recovery/` 及 `archive/harbor-local.lock.json`。

重装相同官方 Harbor 版本后，在本目录应用补丁：

```bash
patch --forward --batch --fuzz=0 -p1 \
  -d "$PWD/.venv-harbor/lib/python3.12/site-packages" \
  -i "$PWD/archive/patches/harbor-0.24.0-terminus-context-recovery.patch"
```

回归测试使用隔离的测试依赖，不修改 Harbor 的依赖环境：

```bash
source scripts/harbor_env.sh
export PIP_CONFIG_FILE=/dev/null
.venv-harbor/bin/python -m pip install --proxy '' --no-cache-dir \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --target .cache/tmp/harbor-context-test-deps \
  -r archive/requirements-harbor-context-tests.lock.txt
PYTHONPATH="$PWD/.cache/tmp/harbor-context-test-deps" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-harbor/bin/python -m pytest -p pytest_asyncio.plugin \
  -o asyncio_mode=auto -o asyncio_default_fixture_loop_scope=function \
  archive/tests/test_terminus_context_recovery.py -q
```

已完成的 seed42/83 项旧评测使用修复前版本，原结果保留。补丁安装阶段未重跑任务；其后另选 9 个被诊断为上下文恢复循环的任务复测，结果独立保存，不改写旧分数。正式对比时三组模型应使用同一修复版本。端口隔离、外部网络、OOM 和判分器 HTTP 超时属于独立问题。

9 题诊断复测保存在 `results/Qwen3.6-35B-A3B/terminal-bench-2.0/context-recovery-retest/`，选择依据和补丁凭据见其 `provenance/selection.json`。首轮运行中三个副本地址失联，2026-10-17 13:13（北京时间）按用户要求全部改用主服务，保留 `attempt1`，将未完成或连接失败的 9 题重新排队为后续 attempt。seed 42、8 并发、每题 4 CPU / 32 GiB、3 小时、采样及 token 预算保持不变。切换前配置、进度快照及原 83 题结果校验值保存在 `provenance/primary-service-switch-20261017/`。

9 题诊断复测于 2026-10-17 16:20（北京时间）结束：通过 2、超时 6、执行错误 1，报告位于该结果目录的 `seed42/report.md`。原后台服务为 `coding-agent-tb2-context-retest-primary.service`，已退出。本轮仅为 9 题诊断子集；后续全量补齐评测单独保存在 `full-seed42-context-fixed/`。

## HLE text-only

固定 test split 的全部 text-only 数据为 2158 题，不使用工具。`bash run_hle.sh` 固定评测 216 题子集；只有显式追加 `--full` 才评测全部 2158 题。结果按模型名保存，全量为 `results/<模型名>/hle/`，子集为 `results/<模型名>/hle-text-only-10pct/`，不使用时间戳 run_id；模型名含 `/` 等路径字符时转换成 `__`。

`run_hle.sh` 是普通 Bash 脚本，固定路径和参数直接写在一条 Python 调用中，自动完成生成、判分与汇总，使用已有 Python 环境执行 `scripts/eval_hle.py`，无需 Harbor、Docker、Apptainer 或 systemd。`VIDEO_JUDGE_KEY` 未设置时，脚本通过 `source /mnt/data/rui.yan/.secrets/skyrouter.env` 加载本机凭据。脚本内直接设置默认模型 `Qwen3.6-35B-A3B` 和现有四台同模型服务，默认每台 32 并发、总计 128，thinking 明确开启。新模型用 `--model` 和 `--base-url` 覆盖，`--base-url` 后直接列出一个或多个以空格分隔的地址，新列表整体替换默认四台地址。服务需已启动，并支持 vLLM 的 `/tokenize` 和 OpenAI 兼容的 `/v1/chat/completions`。模型名须与服务 `/v1/models` 返回值一致。

默认参数：每个模型服务并发 32、temperature=1.0、top_p=1.0、top_k=20、seed=42、thinking=on；总上下文 262144 tokens，输出上限 131072 tokens。每题通过同一副本的 vLLM `/tokenize` 使用相同 chat template 计算输入长度，再发送 `/v1/chat/completions`；实际 `max_tokens=min(131072, 262144-输入tokens, 服务上下文上限-输入tokens)`。不向被测模型发送答案或解析。超时默认 7200 秒，瞬时错误最多重试一次，记录每次尝试。

已完成的 Base 全量运行同时使用原第一台服务和新增的三个 `qwen36-35b` 四卡副本，每台 32 并发，总计 128，最终为 **470/2158 = 21.78%**。`--base-url` 指向原服务，`--replica-manifest` 从模型部署清单的 `replicas` 列表读取通过真实推理检查的新副本；就绪副本自动加入任务队列，待初始化副本暂不接收请求。服务检查由模型目录的 `scripts/watch_service.py --manifest` 执行，部署清单为 `../../logs/qwen36-35b-replicas.json`。每题记录实际 `base_url` 与评测代码 SHA256。若仅使用清单中的新副本，可添加 `--replicas-only`；也可通过重复 `--replica-base-url` 指定固定副本列表。多服务评测须确保各服务为同一模型 checkpoint。`--workers` 始终表示每台生成服务的并发数，判分并发单独保持 32。

新模型全量评测，完成生成后自动由 Skyrouter `gpt-6.1-sol` 判分：

```bash
cd /mnt/data/rui.yan/projects/coding_agent/benchmark
bash run_hle.sh --full \
  --model NEW_MODEL_NAME \
  --base-url http://MODEL_IP:8200/v1
```

判分复用官方 HLE commit `22ed3074b1e7b134bcbc09028d0ba320839b0655` 的 `JUDGE_PROMPT` 和答案结构：判分模型抽取最终答案及置信度，根据标准答案判断等价性，返回 `correct=yes/no`。官方问题提示词也从该固定源码读取。判分默认沿用官方请求的 4096 completion tokens、不指定 temperature，模型换成用户指定的 Skyrouter 路由；这些是本次实验设置，不声明完全复现官方榜单分数。若提供方有固定模型 snapshot，可用 `--judge-model-revision` 记录；未提供时保留为未知，不虚构版本。

结果目录保存 `run.json`（生成配置与数据版本）、`judge.json`（固定判分配置）、`predictions.jsonl`、`grades.jsonl`、`progress.json`、`summary.json` 和 `raw/`（API 原始输出）。密钥仅从环境变量读取，结果只记录变量名。`accuracy_percent` 在全部生成尝试和有效回答判分结束前为 `null`；生成失败计为错误并保留在所选完整任务清单的分母中，判分失败保持待判状态，不当作模型答错。截断、空回答、请求失败分别保留记录，截断计数包含未给出最终答案的失败输出。JSON 文件以唯一暂存文件原子替换，支持进度写入时查看汇总。汇总保存生成与判分 token 用量，以及官方 Wald 95% 区间；不估算未采集的 GPU 成本或 calibration error。

同目录、同参数默认断点续跑，失败任务只有显式添加 `--retry-failed` 才会重试。参数、数据、prompt 或代码发生变化时拒绝混写；需先移动旧结果目录。`--limit N` 只用于调试，结果会明确标记为子集，调试时请另设 `--results-root /tmp/hle-debug`。当前 checkpoint revision、量化等服务端未暴露的信息在记录中明确为未知，正式提交前应补齐。

本次按用户要求从原单服务切换到三个新副本，保留已完成的 148 题。旧配置、旧评测源码、新配置、切换时任务 ID 和已保存结果的哈希均保存在结果目录 `provenance/`，`run.json` 的 `configuration_history` 指向切换凭据。采样、上下文、输出预算、数据及判分协议保持相同，切换前未落盘的请求重新生成。常规重跑仍严格检查配置，不会自动修改历史配置。

随后按用户要求加入原第一台服务，总并发从 96 增至 128，保留当时已完成的 817 题。此轮未修改评测源码，四台服务共用相同的采样与判分设置；切换凭据见 `provenance/deployment-add-primary-20261016T112257.json`。

### HLE 固定 1/10 快速验证集

固定子集为 **216 / 2158 题（10.01%）**，数据文件为 `datasets/hle/text_only_10pct_seed42.jsonl`，清单与分布核对在 `archive/manifests/hle-text-only-10pct-seed42.json`。题目内容与固定版本的 text-only 全集逐行一致，均无图像。

先按 **8 类学科 × 2 种题型**按比例分配名额，同时约束全局题目输入长度分布：长度排名 0–50%、50–90%、90–99%、99–100% 的题目分别抽取 **108、87、19、2** 题。长度来自现有 Base 运行使用相同官方 prompt 与 chat template 得到的 `/tokenize` 计数。配额采用最大余数法，并在学科/题型组内调整以满足两组边际配额；每个分层中用固定 seed=42 的 SHA256 排序取样。抽样仅使用学科、题型、输入长度、ID 和种子；答案正确性与推理长度在选定之后用于核对。

| 学科 | 全集 | 子集 |
| --- | ---: | ---: |
| 数学 | 976 | 98 |
| 计算机/AI | 224 | 23 |
| 生物/医学 | 222 | 22 |
| 物理 | 202 | 21 |
| 人文/社会科学 | 193 | 19 |
| 其他 | 176 | 17 |
| 化学 | 101 | 10 |
| 工程 | 64 | 6 |

子集包含 164 道 exactMatch 和 52 道 multipleChoice。学科占比相对全集的最大偏差为 **0.36 个百分点**，题型为 **0.30 个百分点**，长度分组为 **0.24 个百分点**。输入 token 中位数均为 **207**；平均输入为全集 342、子集 319，平均生成长度为全集 35398、子集 37067 tokens。完整均值、分位数和各分层统计见清单。

已从完成的全量 Base 结果中提取同题基线，保存为 `results/Qwen3.6-35B-A3B/hle-text-only-10pct/baseline.json`：**43/216 = 19.91%**，全量为 **470/2158 = 21.78%**。这是复用原始回答和已有判分得到的分数，未重复调用模型或判分器；文件保留源结果校验值与逐题正确性。该样本上的 Base 分数比全集低 1.87 个百分点，这个偏差不能直接用于修正新模型分数。216 题中每多答对一题约增加 0.46 个百分点，适合快速比较趋势，小幅提升需由全集确认。固定清单以便后续模型使用相同任务，结果标明为诊断子集。

新模型的快速评测命令如下；`MODEL_NAME` 使用该服务 `/v1/models` 返回的模型名，不同 checkpoint 使用不同模型名保留结果。输出按模型优先保存在 `results/<MODEL_NAME>/hle-text-only-10pct/`，采样和预算沿用上述全量设置，并支持断点续跑。以下仅提供命令，本次未启动新模型推理。

```bash
cd /mnt/data/rui.yan/projects/coding_agent/benchmark
bash run_hle.sh \
  --model NEW_MODEL_NAME \
  --base-url http://MODEL_IP:8200/v1
```

多服务直接列在同一个 `--base-url` 后面；例如下面三台服务各 32 并发，总计 96。每台服务应加载相同 checkpoint，输出中保留每题实际请求的地址。

```bash
bash run_hle.sh --model NEW_MODEL_NAME \
  --base-url http://MODEL_IP_1:8200/v1 http://MODEL_IP_2:8200/v1 http://MODEL_IP_3:8200/v1
```

需要调整并发或预算时追加相应选项，例如 `--workers 16`；需要全量时追加 `--full`。默认四个模型地址、判分配置、每台 32 并发、256K 上下文、128K 输出、开启 thinking 和采样设置均直接写在 `run_hle.sh`。查看参数用 `bash run_hle.sh --help`。

重建脚本为 `archive/scripts/prepare_hle_subset.py`，依赖已固定的完整数据及现有 Base 运行，重复执行保持相同任务与产物，拒绝覆盖内容不同的既有固定子集。

离线 HTTP 集成自检：

```bash
.venv-harbor/bin/python -B archive/scripts/test_eval_hle.py
```

### TB2 Apptainer 环境修复与固定 17 题复测（2026-10-18）

继续使用 Harbor 0.24.0 的 Singularity/Apptainer 后端和原有任务镜像。每个 trial 的 Harbor 与容器置于同一个独立网络命名空间，使用本地校验过的 slirp4netns 提供出网；不修改宿主机路由、防火墙或低端口设置。任务独占 localhost，私有网络允许绑定 25/80/8000/8080。执行超时由原 agent/verifier 预算控制，移除适配层额外的 600 秒 HTTP 截断。

Python 3.9 控制服务使用固定 Debian 标准库拆分包和 pip/setuptools wheel 引导，避免在线最新版 get-pip 的兼容问题。终端大段输入分块转发，保留 asciinema 录制；已用实际安装的 TmuxSession 验证 35/75/262 KB Unicode 输入完整性及 Ctrl+C 恢复。判分用 Git 仓库与 C4 分片只在 agent 结束后上传，原测试脚本和断言不变；Git ref 对照上游 API，C4 文件对照固定 revision 的 Git LFS SHA-256。

修复补丁及版本记录见 `archive/patches/harbor-0.24.0-apptainer-environment-repair.patch`、`archive/harbor-local.lock.json`；容器检查和原产物重判证据见 `archive/validation/tb2-environment-repair-20261018/receipt.json`。保存的 headless-terminal 模型代码在健康环境中通过原判分 7/7，未重新调用模型。

固定复测名单为 `archive/manifests/terminal-bench-2.0-environment-retest17.txt`。当前使用主服务、seed42、每题 **8 CPU/32768 MiB，4 并发**，保留原推理参数，ASI/SWE 暂停。此前误配为每题 32 CPU/49152 MiB；2026-10-18 19:53 按用户纠正改回 8 CPU/32 GiB，并重启 4 个未完成项。7 个已结束项及其旧资源记录保留，详细变更见输出目录的 `provenance/resource-limit-change.json` 和 `archive/validation/tb2-resources-8cpu32g-20261018/`。这是包含资源变更的诊断子集，不能直接与官方 89 题五次均值比较。输出独立存入 `results/Qwen3.6-35B-A3B/terminal-bench-2.0/apptainer-environment-retest17/`，每题实际资源见进度中的 `execution_resources` 与各 attempt 配置。

```bash
bash run_tb2_environment_retest.sh --concurrency 4
```

上述命令应仅在当前复测服务未运行时使用。正在运行的服务为 `coding-agent-tb2-envfix.service`，进度见输出目录的 `seed42/progress.json`。当前并发由输出目录的 `control.json` 控制。YouTube 视频 ZCbvyPbhRfA 当前无法直连，已在模型执行前记为环境未就绪，仍保留在 17 题分母中。ASI/SWE 队列目前只等待 TB2 完成，不启动评测；117 条已落盘 ASI 结果完整保留，8 个未完成容器已停止，原工作目录路径保存在 `archive/validation/tb2-priority-20261018/asi-interrupted-containers.json`。TB2 可运行项结束后，`evaluation-queue/priority.json` 的门槛解除，队列按原计划以原生 resume 续跑 ASI，再运行 SWE。

全 89 题的后续统一环境入口为 `run_tb2.sh`，默认输出使用新的 `apptainer-environment-fixed/`，不复用历史结果。

17 题复测的第一批结果中，`configure-git-webserver` 已通过；`count-dataset-tokens` 的模型输出 79566，原判分要求 79586，属于计算口径错误。`build-cython-ext` 虽通过 10/11 个测试，仍因直连 GitHub 失败而改用 PyPI 源码，缺少题目要求的 `.git`。因此追加单题复测，旧结果保留。

单题入口 `bash run_tb2_public_source_retest.sh` 使用原题明确要求的公开 `pyknotid` 0.5.3 仓库缓存，commit 为 `441c807dbec2ee32e1da572e24e58d52a4eb7afa`。只将原 Git URL 指向同一版本的本地副本，agent 仍需自行 clone、修改和编译；未预建 `/app/pyknotid`，判分资源仍在 agent 结束后提供。真实容器中已验证原 clone 命令及 commit，证据位于 `archive/validation/tb2-public-source-repair-20261018/`。

单题结果独立保存在 `apptainer-environment-public-source-retry1/`，服务为 `coding-agent-tb2-public-source.service`。该追加复测已结束，原 Git clone 检查通过，但 NumPy `np.int` 兼容修改不完整，原判分仍为 10/11，因此保留为模型输出失败，不再次作为环境错误重跑。不要在该服务运行期间重复执行单题入口。

后续检查又确认两处环境残留：Apptainer 注入的 `PS1/PROMPT_COMMAND` 使 headless-terminal 等不到 Bash 提示符；Bullseye 安全更新旧包已移到 Debian 官方归档，在线源返回 404。现在仅清理已知的运行时默认提示符，并对 Debian 11 使用签名的 `archive.debian.org/debian-security` 源。使用本轮原模型代码做零模型调用对照，headless 测试从 1/7 变为 6/7，整题仍为 reward 0；QEMU 原判分安装流程已验证可以装齐 curl、expect、uvx 和 pytest。详细证据位于 `archive/validation/tb2-residual-environment-20261018/`，原题 859 个文件保持不变。

上述两题和随后发现的控制进程清理问题，已统一纳入下述新 89 题评测。`coding-agent-tb2-residual.service` 和 `coding-agent-tb2-control.service` 在启动任何原生 trial 前停止，各输出目录的 `superseded.json` 保留替代凭据。qemu-alpine-ssh 的旧运行也更新过私有 APT 源，之后因连续异常模型回复中止并保留完整轨迹。

查看包含全部补跑的 89 题暂计成绩用 `python3 scripts/tb2_environment_status.py`。按最新已结束的指定补跑替换旧结果，无论补跑通过与否；零模型调用的重判不计入成绩。仍保留历史配置差异，不能作为统一设置的 89 题五次均值。

### TB2 请求缓存错误复现与统一 89 题重测（2026-10-18）

主服务日志确认 vLLM 0.28.0、Qwen3.5 架构、MTP 3 和前缀缓存同时启用。Alpine 旧运行在 12:17Z–14:15Z 连续出现 446 个解析错误，正文多数为空，思考字段是无关词句和乱码；旁路采样确认这些是 HTTP 200 的模型服务回复。实际请求的温度、top_p、top_k、thinking 和 seed 正确。对应配置有 [上游同类报告 #53912](https://github.com/vllm-project/vllm/issues/53912)，本地进一步用原始请求做了直接对照：

| 同一上下文、seed42、相同采样，诊断输出上限 2048 | 命中缓存 tokens | 回复 |
| --- | ---: | --- |
| 首次失败前的 35,286-token 上下文，原缓存 | 34,272 | 空正文、乱码思考 |
| 同一上下文，全新 cache_salt | 0 | 正常 JSON 命令 |
| 错误循环中的 56,332-token 上下文，原缓存 | 55,488 | 空正文、乱码思考 |
| 同一上下文，全新 cache_salt | 0 | 正常 JSON 命令 |

证据位于 `archive/validation/tb2-gap-investigation-20261018/cache-comparison/receipt.json`。随后通过真实 Harbor LiteLLM 和 Terminus JSON parser 连续验证两次，均无缓存命中且能解析命令，见 `harbor-cache-integration.json`。两组对照显示缓存命中与异常有明显关联，但每个上下文仅有一组暖/冷缓存对照，温度为 1，主服务仍有并发请求；尚未完成平衡重复实验、关闭 MTP 对照、GDN 状态检查或上游补丁验证。拒绝草稿 token 后的 GDN 快照污染目前是与上游报告一致的假说，不能视为已经闭环的具体根因，隔离缓存属于已验证可恢复这些样本的规避措施；对全部任务分数的影响尚未量化。全 89 题轨迹扫描另见 `all89-parser-audit.json`；普通格式错误和没有结束思考的回复仍需单独判断。

评测客户端新增 `--prefix-cache-policy isolated`，每次 API 调用生成新的 vLLM `cache_salt`，避免复用异常的递归状态。主服务、模型、提示词、采样和预算保持原值；代价是重复上下文需要重新计算。该选项是新的 `run_tb2.sh` 默认设置，补丁和源码哈希已写入 Harbor lock。历史运行中的进程继续使用启动时的客户端代码。

此外，Windows 的 `kill -9 $(pgrep -f 'qemu|nc|python3')` 会误杀容器内的 Python 控制服务。现在该服务使用独立的解释器入口 `hbexec`，普通 Python 清理仍能杀掉任务 Python。真实隔离容器中，原命令在修复前导致断线，修复后控制服务、终端和文件状态均保留，证据见同目录 `control-baseline/` 和 `control-protected/`。控制键处理与核验过的上游一致，未替模型纠正 `C-c\n` 等错误输入。

新完整运行使用 `scripts/run_tb2_fixed89.py`，服务 `coding-agent-tb2-fixed89.service`，结果独立写入 `apptainer-environment-fixed/`。任务集合固定为全部 89 题，`terminal-bench-2.0-fixed89-priority.txt` 将已发现受影响任务排在前面；859 个原题文件和评分断言均未修改。每题 8 CPU/32 GiB、seed42、3 小时、80K 输出/256K 上下文。新任务先使用一个空闲槽，旧 17 题的最后三题结束后逐步提高新队列并发，新旧实际任务总数始终至多 4，分配状态见 `shared-slot-allocation.json`。ASI/SWE 等待门槛已包含这轮完整 89 题。

这轮最终成绩仅来自新的 89 题原生结果，未完成时 `uniform_repaired_full89.accuracy_percent` 留空。旧的暂计 38/89=42.70% 单独保留，诊断重放和原产物重判均不计分。当前用户资源和单 seed 与官方 32 CPU/48 GB、五次平均仍有协议差异，不能预先承诺重现 51.5%。

2026-10-18 23:16 按用户最新指令，`apptainer-environment-fixed/` 保留为中止的诊断运行；重新启动全部 89 题，结果改存 `apptainer-prefix-isolated-full89/`，服务为 `coding-agent-tb2-prefix-isolated89.service`，不复用前一轮结果。入口默认输出也已更新。该轮每请求重新计算上下文，跨请求前缀命中为零，单次生成内部 KV 仍启用；主服务配置未修改。新旧任务合计至多 4 并发、每题 8 CPU/32 GiB，ASI/SWE 继续等待。

用户明确要求 GDN/MTP 问题继续调查和测试，候选修补暂不应用到真实模型，待次日再决定；没有安排自动部署。隔离调查与 13 项 CPU 检查见 `archive/validation/tb2-gdn-cache-research-20261018/REPORT.txt` 和 `investigation-report.json`。其中包含一个“再丢一个 GDN 块”候选的协调器反例、实际接受计数代码的换序反例，以及内存重叠复制的机制演示；这些结果尚不能确认线上唯一根因，也未执行候选的 GPU/模型修补对照。当前模型对照样本的证据边界与后续验证计划已在报告中单列。
