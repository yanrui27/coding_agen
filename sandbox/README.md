# 沙盒控制

在 ACP 纯 CPU Worker 里用 Apptainer 运行任务沙盒，供 Agent RL、评测等调用方远程执行命令、读写文件、使用终端。系统按“调用方 → 控制器 → 宿主”三层组织：调用方按镜像向控制器申请沙盒；控制器在 ACP 上按需建删宿主作业，把沙盒分配到有空位的宿主，并管理租约；宿主是一个 ACP Worker，可同时运行多个沙盒。命令和文件流量由调用方直连宿主，不经过控制器。

```text
调用方（RL rollout / 评测 / MiMo remote_apptainer）
  │ ① POST /sandboxes {image}           ③ 命令、文件、终端直连 <宿主>/sandboxes/<id>
  ▼                                      │
controller.py（第二层，可在开发容器或 ACP 上运行）
  │ ② 选宿主并创建沙盒；不够时经 ACP CLI 提交新宿主作业，空闲宿主删除
  ▼
host_server.py（第三层，每个 ACP CPU Worker 一个，承载 N 个沙盒）
  ├─ 本地镜像基础 rootfs（每种镜像解包一次）
  └─ 每个沙盒：私有可写 rootfs 副本 + apptainer exec 里的 process_agent.py
       ├─ 普通命令（每条一个进程组）
       ├─ 常驻 Python / Bash 会话
       └─ PTY 终端
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `controller.py` | 第二层控制器：租约（创建、续期、释放、到期回收）、宿主选择（优先已解包该镜像的宿主）、按需扩容和空闲缩容、退出时删除自己管理的宿主作业；状态存盘，重启后接管仍存活的宿主和租约 |
| `host_server.py` | 第三层宿主：从共享镜像缓存解包基础 rootfs，为每个沙盒复制私有副本、启动 Apptainer 和 `process_agent.py`，在沙盒外提供 HTTP API；租约到期、空闲、到寿命或 SIGTERM 时清理 |
| `process_agent.py` | 沙盒内的进程管理器，经 stdin/stdout 与宿主通信；用户代码只在它创建的子进程里运行 |
| `client.py` | 调用方用的客户端库和命令行（申请、续期、释放、执行命令、读写文件） |
| `images.py` | 共享镜像缓存：登记已有 rootfs（tar、目录或 SIF），或从 Docker Hub 镜像站拉取镜像层 |
| `run_controller.sh` | 控制器常驻入口，参数写在脚本里 |
| `check_pool.py` | 整体检查：并发申请、隔离、token 权限、续期、释放、租约到期 |
| `check_control.py` | 单个沙盒的 17 项控制检查（真实进程、常驻会话、取消/超时、崩溃恢复、冻结后重启），会冻结并重启被测沙盒 |
| `build_apptainer.py` | 从官方 PPA / Ubuntu Jammy 包构建 `runtime/apptainer-1.5.4-jammy` |
| `runtime/` | Apptainer 工具包、镜像缓存 `images/`、控制器状态 `controller/`，不进 git |

## 用法

启动控制器（在本开发容器上验证过；ACP Worker 也能访问 ACP API，可以作为 CPU 作业运行，但这种部署还没实测）：

```bash
nohup bash /mnt/data/rui.yan/projects/coding_agent/sandbox/run_controller.sh &
```

默认配置：宿主规格 `N3lS.Ii.I60.32c128g`（32 vCPU / 128 GiB，不含 GPU），每台 8 个沙盒，最多 8 台（64 路），空闲 600 秒的宿主会被删除。控制器地址和 token 写在 `runtime/controller/controller.private.json`（权限 0600），日志在同目录的 `controller.log`，每台宿主的 payload 和日志在 `runtime/controller/hosts/<作业名>/`。停止控制器用 SIGTERM，会删除它管理的全部宿主作业。

调用方（Python）：

```python
import sys
sys.path.insert(0, '/mnt/data/rui.yan/projects/coding_agent/sandbox')
from client import Controller

controller = Controller.from_file('/mnt/data/rui.yan/projects/coding_agent/sandbox/runtime/controller/controller.private.json')
with controller.create('docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001661', ttl_seconds=3600) as sandbox:
    sandbox.write_file('/tmp/test.patch', patch_bytes)
    result = sandbox.run('git apply /tmp/test.patch && python -m pytest -q', cwd='/workspace/repo', timeout=600)
    print(result['returncode'], result['stdout'])
# 离开 with 即释放；长任务定期 sandbox.renew(3600)
```

命令行，以及接 MiMo `remote_apptainer` 后端：

```bash
cd /mnt/data/rui.yan/projects/coding_agent/sandbox
CONTROLLER=runtime/controller/controller.private.json
python3 client.py --controller $CONTROLLER status
python3 client.py --controller $CONTROLLER create --image IMAGE --endpoint-out /path/endpoint.private.json
python3 client.py --controller $CONTROLLER release SANDBOX_ID
```

`create` 写出的 `{url, token}` 文件就是 MiMo `remote_apptainer` 后端的 `endpoint_file`（mimo-rl 的 `configs/code-acp-apptainer.yaml`），也可以直接交给 `check_control.py`。[benchmark/scripts/harbor_acp_sandbox.py](../benchmark/scripts/harbor_acp_sandbox.py) 用同一个客户端实现了 Harbor 的 `AcpSandboxEnvironment`，通过 `environment.kwargs.controller_file` 指向控制器凭据文件；题目镜像需先登记到控制器缓存。

镜像要先进共享缓存（`runtime/images/<镜像名>/image.json`，记录镜像名、环境变量和各层路径）：

```bash
python3 images.py add --image docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001661 \
  --rootfs /mnt/data/rui.yan/projects/mimo-rl/.cache/mimo-code-001661.rootfs.tar \
  --image-env PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin --image-env LANG=C.UTF-8
python3 images.py add --image alexgshaw/adaptive-rejection-sampler:20251031 \
  --rootfs ../benchmark/.cache/harbor-images/alexgshaw_adaptive-rejection-sampler_20251031.sif   # Env 取自 SIF
python3 images.py pull --image docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-000001   # 经镜像站下载
python3 images.py list
```

目前缓存里有 `format-code-task-001661`（MiMo Code 真实任务，docker export 的 1.1 GB tar）、`local/compile-compcert:tb2`（85 MB 的 TB2 轻量 rootfs，没有 Python），以及 TB2 全部 89 题的镜像（直接指向 `benchmark/.cache/harbor-images` 里的 SIF；宿主用工具包里的 unsquashfs 按 SIF 头记录的偏移解出主分区）。不在缓存里的镜像，控制器直接返回 422。

## HTTP API

所有请求带 `Authorization: Bearer <token>`，POST 请求体为 JSON（控制器不超过 64 KiB，宿主不超过 4 MiB）。

控制器（控制器 token）：

| 接口 | 作用 |
| --- | --- |
| `POST /sandboxes` | `image`、`ttl_seconds`（默认 3600）、`wait_seconds`（默认 900，等扩容的最长时间）、`label`、`binds`（可选，`[{source, target}]` 只读挂载，`source` 须在控制器 `--bind-root` 之下，如 TB2 的 `.cache/harbor-assets`）；返回 `sandbox_id`、`url`（`<宿主>/sandboxes/<id>`）、`token`、`host`、`wait_seconds`、`create_seconds` |
| `POST /sandboxes/{id}/renew` | `ttl_seconds`，续租 |
| `DELETE /sandboxes/{id}` | 释放：停止沙盒内全部进程并删除私有 rootfs |
| `GET /sandboxes/{id}`、`GET /status`、`GET /health` | 租约、宿主和等待队列状态 |

沙盒（沙盒自己的 token，路径前缀 `<宿主>/sandboxes/<id>`，与原单沙盒服务的接口相同）：

| 接口 | 作用 |
| --- | --- |
| `GET /health` | 由沙盒外直接响应，不进入沙盒；含租约剩余时间 |
| `POST /commands` | `command`、`cwd`、`env`、`timeout_seconds`；异步启动，返回 `job_id` |
| `GET /jobs/{id}`、`POST /jobs/{id}/cancel` | 状态、stdout/stderr 末尾 64 KiB 及其总字节数和沙盒内路径（更长的输出用文件接口读全）、退出码；取消即杀进程组 |
| `POST /sessions`、`POST /sessions/{id}/execute` | 常驻 Python / Bash 会话，变量、cwd、环境变量跨轮保留 |
| `POST /terminals`、`GET /terminals/{id}?offset=`、`POST /terminals/{id}/write`、`POST /terminals/{id}/close` | 真实 PTY，支持增量读取和 Ctrl-C |
| `GET /files/read?path=&offset=`、`POST /files/write` | 按最多 1 MiB 分块读写二进制文件 |
| `POST /reset` | 重启该沙盒（杀掉整个 PID 命名空间），私有 rootfs 保留 |

宿主自身的接口（`GET /health`、`GET|POST /sandboxes`、`DELETE /sandboxes/{id}`、`POST /sandboxes/{id}/renew`）只接受宿主 token，由控制器使用；沙盒 token 访问其他沙盒或宿主接口返回 401/403。

## 生命周期与回收

- 租约：控制器按 `ttl_seconds` 回收不续期的沙盒；宿主上同一沙盒多留 60 秒作为兜底。
- 宿主：控制器每 3 秒检查一次。新宿主写出 `endpoint.private.json` 并通过 `/health` 后才分配沙盒；连续 3 次健康检查失败、或 ACP 状态已结束的宿主会被删除，其上的租约一并作废。启动超过 1200 秒仍未就绪也会删除。
- 控制器崩溃：宿主和沙盒照常运行；重启控制器（同一 `--state-dir`）会接管它们。控制器一直不回来时，宿主在租约全部到期、且 900 秒没有收到任何请求后自行退出，ACP 作业随之结束。
- 孤儿沙盒：宿主上存在、控制器不认识、且已创建超过 120 秒的沙盒会被删除。

## ACP 上的限制与做法

| 限制 | 做法 |
| --- | --- |
| Worker 镜像为 Ubuntu 22.04 / GLIBC 2.35，benchmark 自带的 Apptainer 需要 2.38 | 使用 Jammy 版 Apptainer 1.5.4 工具包 |
| 没有 `/dev/fuse`，不能挂 SIF 和 overlay | 镜像解包成目录，每个沙盒复制一份到 Worker 本地盘，用 `--writable` |
| 拒绝新挂 procfs | `--no-mount proc --no-mount sys`；沙盒内没有 `/proc`，`ps` 等工具不可用 |
| root 没有 subuid 映射 | 宿主启动时执行 `apptainer config fakeroot --add root` |
| `/sys/fs/cgroup` 只读 | 无法给单个沙盒限 CPU/内存，只按槽位数控制每台宿主的沙盒数 |
| Docker Hub 主站不可达 | `images.py pull` 走镜像站（默认 dockerproxy.net）；本地盘 200 GB |

## Apptainer 工具包

`runtime/apptainer-1.5.4-jammy` 复制自 `mimo-rl/tools/apptainer-1.5.4-jammy`（在 ACP 上验证过的那份），只把 `etc/apptainer/apptainer.conf` 里的 `binary path` 改成了本目录。重建要在 GLIBC ≥ 2.38 的主机上跑 `build_apptainer.py`，当前 CCI 开发容器是 2.31，不能重建，也不能直接运行宿主（本机验证用 `claude_workspace/2026-10-19_沙盒控制整理/scripts/fake-apptainer` 替身）。

## 验证情况

2026-10-20，记录在 `claude_workspace/2026-10-20_沙盒控制系统/`：

- 本机（替身 apptainer，静态宿主 3 槽位）：`check_pool.py` 7 项通过；经控制器分配的沙盒上，MiMo `remote_apptainer` 转发 6 项和 `check_control.py` 17 项通过。
- ACP（控制器在本开发容器，宿主 `N3lS.Ii.I60.8c32g`、每台 3 槽位、最多 2 台）：一次申请 5 个沙盒，控制器从 0 台扩到 2 台（提交到就绪 11 秒和 23 秒），5 个沙盒全部在 34 秒内建好，`check_pool.py` 7 项通过；真实 Apptainer 上 MiMo 转发 6 项、控制检查 17 项通过。宿主首次用到某镜像时建沙盒约 6–9 秒（其中解包 4–5 秒），复用后约 2.5 秒。kill -9 控制器再重启后接管了宿主和租约，原沙盒继续可用；空闲 180 秒的宿主被删除；带活跃沙盒时 SIGTERM 控制器，宿主作业被删除。三台宿主作业最终都返回 404。

更早的单沙盒验证（Codex 2026-10-16 至 17 用 mimo-rl 原型、2026-10-19 整理后复测）见 `claude_workspace/2026-10-19_沙盒控制整理/` 和 `mimo-rl/manifests/acp-*-result.json`。

## 未完成

- `images.py pull` 还没有完整拉取过一个镜像。镜像站单连接实测 0.06–0.7 MB/s，一个 MiMo Code 镜像约 400 MB；`pull` 会多连接分段下载并校验 sha256，但实际速度未测。
- 单个沙盒没有 CPU/内存硬限制，失控任务会影响同宿主的其他沙盒。
- 没有测过默认配置（32c128g、8 槽位）和 64 路并发；控制器部署在 ACP 上未实测。
- 沙盒内缺少 `/proc` 对更多任务的影响尚未系统验证。
