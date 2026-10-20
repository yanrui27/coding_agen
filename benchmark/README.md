# Benchmark 评测

四套评测（Terminal-Bench 2.0、HLE、ASI-Bench、SWE-bench Pro）的数据、官方评测器、运行脚本、依赖环境和结果都在本目录。各轮评测、修复和诊断按时间顺序的完整记录见 [archive/HISTORY.md](archive/HISTORY.md)；本文只写当前状态和用法。

```text
benchmark/
├── run_tb2.sh  run_hle.sh  run_asi.sh  run_swe.sh   各评测集入口，参数直接写在脚本里
├── scripts/       入口调用的评测代码，以及镜像准备、后台启停、数据校验等维护脚本
├── tests/         离线测试
├── datasets/      四套固定版本数据
├── third_party/   官方评测代码
├── results/<模型名>/<评测集>/   结果
├── archive/       版本锁、任务清单、补丁、验证证据、历史记录；tb2-diagnostics/ 是已完成的一次性复测和诊断脚本
├── .venv-harbor/  Harbor 0.24.0 与 Terminus-2
├── .sandbox-tools/apptainer/  Apptainer 1.5.4
└── .cache/        SIF 镜像、SWE/ASI 的 Docker 存储盘、判分资源、临时文件
```

## 当前状态（2026-10-20）

被测模型均为 `Qwen3.6-35B-A3B`，主服务为 `http://10.119.27.14:8200/v1`。

| 评测集 | 状态 | 结果 | 结果目录（`results/Qwen3.6-35B-A3B/` 下） |
| --- | --- | --- | --- |
| Terminal-Bench 2.0（89 题） | 暂停，2026-10-19 由用户停止 | 最新一轮完成 79/89：通过 37、失败 23、超时 17、环境错误 2，还剩 10 题未跑；最终通过率尚未产生 | `terminal-bench-2.0/apptainer-prefix-isolated-full89/` |
| HLE text-only | 完成 | 全量 2158 题 470 题正确，21.78%；固定 216 题子集上的同题基线 43/216 = 19.91% | `hle/`、`hle-text-only-10pct/` |
| ASI-Bench seed42 | 暂停 | 240 个条件中已跑 117 个（完成 97、超时 13、失败 7）；正式分数要由 ASI 官方平台用私有参考答案计算 | `asi-bench/` |
| SWE-bench Pro V2 HARD-51 | 未开始 | 51 个镜像已准备并核验 | — |

TB2 → ASI → SWE 的后台接续队列（`coding-agent-eval-queue.service`）同样已在 2026-10-19 停止，不会自动续跑。

## 运行环境

需要 Linux、Python 3.12，以及本机 systemd/cgroup 权限（目前在 real-241 上运行）；HLE 只需 Python 和 HTTP。所有下载都直连，入口会清除代理变量。

TB2 使用 Harbor 0.24.0 + Terminus-2，Apptainer 作为 `singularity` 后端。官方版本之上打了若干本地补丁（`archive/patches/`），涉及上下文恢复、Apptainer 环境修复、控制进程保护，以及每次请求隔离 vLLM 前缀缓存（`--prefix-cache-policy isolated`）。补丁和安装后源码的哈希记录在 `archive/harbor-local.lock.json`，数据版本记录在 `archive/benchmarks.lock.json`，任务清单在 `archive/manifests/`。

SWE 和 ASI 各自使用一个专用的 dockerd，数据目录分别是 `.cache/docker-images/` 下以 loop 方式挂载的 ext4 磁盘文件。挂载点和 dockerd 配置由入口脚本按相对路径自动生成。

环境自检和数据校验：

```bash
cd /mnt/data/rui.yan/projects/coding_agent/benchmark
bash scripts/manage_tb2.sh harbor --version
bash scripts/verify_data.sh
```

## 用法

以下命令都在本目录执行。

### Terminal-Bench 2.0

`run_tb2.sh` 写明了全部参数：89 题，seed 42，4 题并发，每题 8 CPU / 32 GiB、Agent 限时 3 小时，输出上限 80K、上下文 256K，temperature 1.0、top_p 0.95、top_k 20，开启 thinking，前缀缓存隔离。结果写入 `apptainer-prefix-isolated-full89/`。在同一目录用相同配置再次运行时，已完成的任务会跳过，可用于续跑剩下的 10 题。

```bash
bash run_tb2.sh --print-config        # 只看最终配置
bash scripts/manage_tb2.sh start      # 后台运行（也可直接 bash run_tb2.sh 前台运行）
bash scripts/manage_tb2.sh status     # 后台服务状态
bash scripts/manage_tb2.sh summary    # 当前汇总
bash scripts/manage_tb2.sh stop
bash scripts/manage_tb2.sh images status   # SIF 镜像准备：start / status / stop / repair
```

### HLE text-only

`run_hle.sh` 默认评测固定的 216 题子集，加 `--full` 评测全部 2158 题。判分使用 Skyrouter 上的 `gpt-6.1-sol`，提示词沿用官方 HLE；如果没有设置 `VIDEO_JUDGE_KEY`，脚本会自动加载本机凭据。脚本里写的是四台服务地址，换模型时用 `--model` 和 `--base-url` 覆盖。

```bash
bash run_hle.sh --model NEW_MODEL_NAME --base-url http://MODEL_IP:8200/v1          # 216 题
bash run_hle.sh --full --model NEW_MODEL_NAME --base-url http://MODEL_IP:8200/v1   # 全量
```

### ASI-Bench seed42

默认参数：60 题 × B1–B4 共 240 个条件，8 个沙盒并发，每个条件 3 小时、4 CPU / 32 GiB，Agent 为 OpenCode 1.17.15。

```bash
bash run_asi.sh --print-config
bash run_asi.sh --check                                      # 检查数据、镜像和模型服务，不跑任务
bash run_asi.sh --resume results/Qwen3.6-35B-A3B/asi-bench   # 续跑尚无结果的条件
bash scripts/manage_asi.sh cli report --results-dir results/Qwen3.6-35B-A3B/asi-bench
bash scripts/manage_asi.sh cli submit --results-dir results/Qwen3.6-35B-A3B/asi-bench \
  --benchmark-repo Apexintelligence-AI/ASI-Bench-seed42 --no-upload
bash scripts/manage_asi.sh docker images                     # 也支持 prepare-images
```

### SWE-bench Pro V2 HARD-51

Agent 为 mini-swe-agent 2.4.6，在宿主机上运行，任务命令在断网的 4 CPU / 32 GiB 容器里执行。结束后把补丁放到全新容器里，用官方测试判分。每题 Agent 和判分各有 3000 秒预算，8 题并发。

```bash
bash run_swe.sh check     # 核对 51 题、镜像和依赖
bash run_swe.sh smoke     # 不调用模型的隔离、补丁捕获与回放检查
bash run_swe.sh run
bash run_swe.sh status
```

### 评测队列与测试

```bash
.venv-harbor/bin/python scripts/run_eval_queue.py --status
.venv-harbor/bin/python -B tests/test_eval_hle.py
.venv-harbor/bin/python -B tests/test_eval_asi.py
.venv-harbor/bin/python -B tests/test_evaluation_queue.py
```

Terminus 上下文恢复补丁的回归测试需要单独安装测试依赖，命令见 `archive/HISTORY.md` 的“上下文恢复补丁”一节，测试文件现在是 `tests/test_terminus_context_recovery.py`。

## TB2 结果目录索引

均位于 `results/Qwen3.6-35B-A3B/terminal-bench-2.0/`，按时间排序。只有最后一行是当前主线，其余都是历史或诊断结果，分数不能直接互相比较。

| 目录 | 范围与配置 | 结果 | 说明 |
| --- | --- | --- | --- |
| 根目录 `seed42/` | 83 题子集，4 CPU / 32 GiB，修复前的 Agent | 31/83 = 37.35% | 最早的单种子评测，不算完整成绩 |
| `context-recovery-retest/` | 9 题，上下文恢复补丁诊断 | 通过 2、超时 6、错误 1 | 诊断子集 |
| `full-seed42-context-fixed/` | 89 题：沿用 73 题旧结果，新跑 16 题 | 通过 34、失败 27、超时 19、错误 9 | 混合了两个 Agent 版本，未隔离前缀缓存 |
| `apptainer-environment-retest17/` | 17 题环境修复复测，8 CPU / 32 GiB（其中 7 题按误配的 32 CPU / 48 GiB 跑完） | 通过 4、失败 7、超时 2、错误 4 | 诊断子集 |
| `apptainer-environment-public-source-retry1/` | 单题 build-cython-ext | 失败 | 诊断 |
| `apptainer-environment-residual-retest2/`、`apptainer-control-cleanup-retest2/` | — | 一题都没跑 | 被下一行取代 |
| `apptainer-environment-fixed/` | 89 题统一修复版 | 完成 2 题后中止 | 被下一行取代 |
| `apptainer-prefix-isolated-full89/` | 89 题，8 CPU / 32 GiB，前缀缓存隔离 | 79/89 已完成，通过 37 | 当前主线，已暂停 |

Qwen 官方模型卡报告的 TB2.0 成绩是 51.5%，条件是 32 CPU / 48 GB、5 次运行取平均；本地评测在硬件和运行次数上都与之不同。

## 已知问题

- vLLM 0.28 在 MTP、异步调度与前缀缓存同时开启时，命中缓存的请求可能读到错位的 GDN 状态，输出乱码。TB2 因此改为每次请求隔离缓存。根因分析和 GPU 验证见 `claude_workspace/2026-10-19_命中缓存乱码调查/`，早期调查见 `archive/validation/tb2-gdn-cache-research-20261018/`。
- 本目录 2026-10-19 从 `data/benchmark/` 移到这里。results、archive 和日志里记录的仍是旧绝对路径，这些是历史记录，没有改写。SWE/ASI 的 dockerd 如果还挂在旧路径上，需要先停掉 dockerd 并卸载旧挂载，入口脚本才能按新路径重新挂载。
