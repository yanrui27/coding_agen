# 沙盒控制

在 ACP 纯 CPU Worker 里用 Apptainer 起任务沙盒，由外部通过 HTTP 控制。按“Agent → 调度 → 执行”三层划分，本目录是第三层执行层：一个控制服务对应一个沙盒，Agent 发来的命令、文件读写、终端输入都在沙盒里以真实 Linux 进程执行，结果原路返回。第二层调度（自动建 ACP 作业、按 `sandbox_id` 路由、回收）还没有实现；第一层目前是 mimo-rl 里的 MiMo 环境后端 `remote_apptainer`。

代码由 Codex 在 `/mnt/data/rui.yan/projects/mimo-rl/scripts/` 的原型整理而来，原型及其 ACP 验证记录保留在 mimo-rl 未动。

## 文件

| 文件 | 作用 |
| --- | --- |
| `control_server.py` | 在 Worker 上运行：把 rootfs 复制到本地私有目录，用 Apptainer 启动 `process_agent.py`，在沙盒外提供 HTTP API；TTL 到期或收到 SIGTERM 后停沙盒并清理 |
| `process_agent.py` | 沙盒内的进程管理器，经 stdin/stdout 与控制服务通信；用户代码只在它创建的子进程里运行 |
| `check_control.py` | 17 项控制检查：真实进程、常驻会话、取消/超时、崩溃恢复、冻结后重启。会冻结并重启被测沙盒，只对测试沙盒使用 |
| `build_apptainer.py` | 从官方 PPA / Ubuntu Jammy 包构建 `runtime/apptainer-1.5.4-jammy` |
| `runtime/` | Apptainer 工具包和包缓存，不进 git |

```text
ACP Worker
├─ control_server.py（沙盒外）：HTTP API、令牌鉴权、TTL、重启沙盒
└─ apptainer exec（私有可写 rootfs，不挂 /proc、/sys）
   └─ process_agent.py：只管理进程，不执行用户代码
      ├─ 普通命令（每条一个进程组）
      ├─ 常驻 Python / Bash 会话
      └─ PTY 终端
```

任务卡死或崩溃时控制服务仍能响应；沙盒内的进程管理器被冻结时，`/health` 仍可用，`/reset` 会杀掉整个 PID 命名空间后重新启动，私有 rootfs 里的文件保留。

## ACP 上的限制与做法

| 限制 | 做法 |
| --- | --- |
| Worker 镜像为 Ubuntu 22.04 / GLIBC 2.35，benchmark 自带的 Apptainer 需要 2.38 | 使用 Jammy 版 Apptainer 1.5.4 工具包 |
| 没有 `/dev/fuse`，不能挂 SIF 和 overlay | rootfs 预先解包成目录或 tar，复制到 Worker 本地 `/tmp`，用 `--writable` |
| 拒绝新挂 procfs | `--no-mount proc --no-mount sys`；沙盒内没有 `/proc`，`ps` 等工具不可用，依赖 `/proc` 的任务兼容性尚未系统验证 |
| root 没有 subuid 映射 | 启动时执行 `apptainer config fakeroot --add root` |
| 在共享盘上解包 GB 级镜像太慢 | 一律解到 Worker 本地盘 |

## 用法

ACP 作业（Codex 用的是 `N3lS.Ii.I60.8c32g`，资源请求不含 GPU；镜像为 `registry.cn-sh-01.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image`）的启动命令：

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*" no_proxy="*" PYTHONUNBUFFERED=1
python3 /mnt/data/rui.yan/projects/coding_agent/sandbox/control_server.py \
  --output-dir <每个作业唯一的输出目录> \
  --rootfs /mnt/data/rui.yan/projects/mimo-rl/.cache/mimo-code-001661.rootfs.tar \
  --image-manifest <镜像清单 JSON> \
  --ttl-seconds 7200 > <输出目录>/worker.log 2>&1
```

输出目录里会出现 `endpoint.json`（地址、rootfs 标记、agent 信息）和 `endpoint.private.json`（URL 和令牌，权限 0600，退出时删除）。镜像清单需要包含 `image`（任务镜像名，MiMo 后端会核对）和 `image_env`（镜像的 ENV 列表），格式见 `/mnt/data/rui.yan/projects/mimo-rl/outputs/mimo-agent-cpu8-20261017-130834/image-manifest.json`。释放沙盒就删除 ACP 作业：控制服务收到 SIGTERM 后会停沙盒并清理。

检查一个刚起好的测试沙盒：

```bash
python3 check_control.py --endpoint-file <输出目录>/endpoint.private.json --output <结果>.json
```

MiMo Agent 接入：后端在 `/mnt/data/rui.yan/projects/mimo-rl/mimoagent/src/mimoagent/environments/remote_apptainer.py`，配置为 mimo-rl 的 `configs/code-acp-apptainer.yaml`，把环境变量 `MIMO_SANDBOX_ENDPOINT_FILE` 指向 `endpoint.private.json` 即可。

## HTTP API

所有请求都要带 `Authorization: Bearer <token>`，POST 请求体为 JSON（不超过 64 KiB）。

| 接口 | 作用 |
| --- | --- |
| `GET /health` | 由沙盒外直接响应，不进入沙盒 |
| `POST /commands` | `command`、`cwd`、`env`、`timeout_seconds`；异步启动，返回 `job_id` |
| `GET /jobs/{id}` | 状态、stdout/stderr 末尾 64 KiB、退出码、进程是否存活 |
| `POST /jobs/{id}/cancel` | 杀掉对应进程组 |
| `POST /sessions`，`POST /sessions/{id}/execute` | 常驻 Python / Bash 会话，变量、cwd、环境变量跨轮保留 |
| `POST /terminals`，`GET /terminals/{id}?offset=`，`POST /terminals/{id}/write`，`POST /terminals/{id}/close` | 真实 PTY，支持增量读取输出和 Ctrl-C |
| `GET /files/read?path=&offset=`，`POST /files/write` | 按 32 KiB 分块读写二进制文件 |
| `POST /reset` | 重启整个沙盒，私有 rootfs 保留 |

## rootfs 与工具包

目前的 rootfs 来自一台有 Docker 的机器：对任务镜像做 `docker export`，得到 tar 包，例如 `mimo-rl/.cache/mimo-code-001661.rootfs.tar`（1.14 GB）。轻量测试可以用 `mimo-rl/.cache/acp-apptainer-rootfs/compile-compcert`（85 MB，没有 Python）。

`runtime/apptainer-1.5.4-jammy` 是从 `mimo-rl/tools/apptainer-1.5.4-jammy`（即在 ACP 上验证过的那份）复制来的，只把 `etc/apptainer/apptainer.conf` 里的 `binary path` 改成了本目录。重建要在 GLIBC ≥ 2.38 的主机上跑 `build_apptainer.py`，因为 benchmark 的 patchelf 需要 2.38；当前 CCI 开发容器是 2.31，不能用来重建。

## 验证情况

- ACP 实测（Codex，2026-10-16 至 17，用 mimo-rl 原型代码，8 vCPU / 32 GiB 纯 CPU Worker）：17 项控制检查通过；在真实任务 `format-code-task-001661` 上，原生 MiMo Agent 经远程接口完成 18 次模型调用、26 次工具调用，隐藏测试 8 通过、1 失败（reward=0，失败原因是模型改错了逻辑）。证据在 `mimo-rl/manifests/acp-*-result.json`。
- 整理后的本机验证（2026-10-19）：本机无法运行 Jammy 版 Apptainer，改用一个用 unshare（用户、挂载、PID 命名空间）加 chroot 模拟的替身 apptainer。轻量 rootfs 和真实任务 rootfs 上，17 项控制检查都通过；mimo-rl 现有 `remote_apptainer` 后端的 6 项工具转发检查通过，说明 HTTP 接口兼容；TTL 到期和 SIGTERM 两种退出方式下，私有 rootfs、私有 endpoint 和沙盒进程都清理干净。记录在 `claude_workspace/2026-10-19_沙盒控制整理/`。
- 整理后的 ACP 复测（2026-10-19，作业 `sandbox-control-cpu8-20261019-113943`，`N3lS.Ii.I60.8c32g`，不含 GPU）：在真实任务 rootfs 上，Worker 提交后约 20 秒就绪；从 CCI 开发容器外部访问，MiMo 工具转发 6 项和控制检查 17 项全部通过，/health 最长约 1.2 ms。删除作业后，控制服务以 SIGTERM 退出，私有 endpoint 已删除，API 返回 404。记录在同一工作区的 `results/acp-sandbox-control-cpu8-20261019-113943/`。

## 相对原型的改动

| 现文件 | mimo-rl 原型 |
| --- | --- |
| `control_server.py` | `probe_acp_supervised.py` 的环境准备 + `supervised_probe_server.py`，合成一个进程；去掉了内置检查和“等待外部客户端”的探针逻辑 |
| `process_agent.py` | `sandbox_process_agent.py` |
| `check_control.py` | `check_supervised_control.py` |
| `build_apptainer.py` | `prepare_acp_apptainer.py` |

HTTP 接口和 `/health` 返回字段保持不变，mimo-rl 的 MiMo 后端可以直接连。行为上的改动有这些：

- 端口改为直接绑定 0 号端口，令牌不再写入磁盘上的配置文件。
- 处理 SIGTERM，TTL 默认 7200 秒。
- 沙盒内路径改名为 `/opt/sandbox-control`、`/opt/sandbox-python`，rootfs 标记前缀改为 `sandbox-`。
- 终端子进程在 exec 之前把 SIGINT/SIGQUIT 恢复为默认处理：控制服务若以 `nohup … &` 启动，原来会继承“忽略 SIGINT”，导致 Ctrl-C 失效。
- 构建脚本在 patchelf 跑不起来时直接报错，不再静默跳过 rpath 修补。

没有迁移的原型：早期能力探针 `probe_acp_apptainer.py`、`run_acp_apptainer_checks.sh`，以及在同一进程里执行用户代码的有状态原型 `probe_acp_stateful.py`、`stateful_probe_server.py`、`check_stateful_session.py`（已被独立进程版取代）。

## 未完成

- 第二层调度：自动创建和删除 ACP 作业、`sandbox_id` 到 endpoint 的路由、租约和回收。
- 一个 Worker 承载多个沙盒，以及每个沙盒的 CPU 和内存限制。
- 镜像供给：2,698 个 Code 任务每题一个镜像（单个约 1 GB），目前只能在有 Docker 的机器上逐个导出。
- 缺少 `/proc` 对更多任务的影响、64 路并发、接入真实 RL 训练。
