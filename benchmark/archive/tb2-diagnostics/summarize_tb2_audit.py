"""Attach reviewed causal assessments to existing TB2 evidence; never rescore."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "benchmark/results/Qwen3.6-35B-A3B/terminal-bench-2.0"
OUT = BASE / "full-seed42-context-fixed/provenance/failure-root-cause-audit-20261018"

# These mean observed blockers, not a claim that an otherwise correct answer
# would have passed. Final solution quality is unknown when grading never ran.
ENVIRONMENT = {
    "qemu-startup": ("bootstrap_python", "Python 3.9 无法运行 Harbor 下载的最新版 get-pip.py；三次启动均失败，模型尚未开始执行。"),
    "qemu-alpine-ssh": ("bootstrap_python", "Python 3.9 与最新版 get-pip.py 不兼容；模型尚未开始执行。"),
    "count-dataset-tokens": ("memory_cgroup_oom", "内核明确记录本题 32 GiB cgroup OOM，杀死 python3，anon-rss=32409256 kB；48 GB 下能否完成未知。"),
    "configure-git-webserver": ("network_namespace", "共享宿主网络，8080 持续 Address already in use；模型看见宿主 nginx。最终环境 exec 服务断连的直接触发点未确认。"),
    "pypi-server": ("network_namespace", "必须使用的 8080 被共享网络中已有监听占用，服务启动失败；最终 Harbor exec 断连触发点未确认。"),
    "mailman": ("network_namespace", "SMTP 127.0.0.1:25 bind 返回 EACCES；fakeroot 未提供任务所需的低端口权限。最终 exec 断连触发点未确认。"),
    "install-windows-3.11": ("network_namespace", "nginx 无法绑定任务要求的 80 端口，返回 Permission denied；宿主监听同时可见。另有 QEMU 长时间执行，不能全部归于端口。"),
    "headless-terminal": ("network_namespace", "唯一失败测试访问 localhost:8000 返回 404。保存的同一份实现，在共享网络复现端口冲突/404，在独立网络命名空间得到 200 和预期内容；没有模型调用。不是全套官方重判。"),
    "build-cython-ext": ("grader_dependency", "11 测试通过 10；唯一失败项在 git clone 后缺少临时 tests 目录，pytest 返回 4，未实际运行该仓库测试。clone 的 stderr 被判分脚本覆盖，具体下载失败原因未保留。"),
    "fix-ocaml-gc": ("grader_dependency", "grader 的 git clone 因 GnuTLS/TLS 失败，脚本仍删除原 testsuite，复制缺失目录失败，随后用空 tests.txt 判 0；此前 agent 实际运行出现 40 tests passed。尚未健康环境全量重判。"),
    "reshard-c4-data": ("grader_dependency", "grader setup 下载 unseen C4 shard 00009 时 Network is unreachable，仅有 shard 00000 缓存，fixture 失败，压缩/解压方案未进入正式测试。"),
    "torch-pipeline-parallelism": ("grader_dependency", "grader 安装 torch 依赖时 cudnn wheel 下载/解压网络超时，无 ctrf 测试记录。不是矩阵计算断言失败。"),
    "query-optimize": ("exec_http_timeout", "官方任务 verifier budget 为 900 秒，但 Singularity exec 默认 HTTP timeout=600 秒，异常链明确是 httpx.ReadTimeout -> HTTP request timed out after 600 seconds，被重包装成 900 秒 verifier timeout；判分只进入第一项。"),
    "extract-moves-from-video": ("external_required_data", "任务所需 YouTube 视频不可达，yt-dlp 多次 Network is unreachable，未能取得输入视频。"),
}
RECORDING = {
    "feal-differential-cryptanalysis": "step 7 一次发送 34513 字符 heredoc；asciinema 2.2.0 的输入和输出记录在该大段输入中途停止，后续长时间无法恢复。",
    "make-mips-interpreter": "step 13 一次发送 70210 字符 heredoc；asciinema 2.2.0 记录在输入中途停止，后续数百步仍停在同一终端画面。",
    "regex-chess": "step 8 一次发送 74915 字符 heredoc；asciinema 2.2.0 记录在输入中途停止，此后反复发送 terminator 也无效。",
}
OUTPUT_FAILURES = {
    "adaptive-rejection-sampler": "接口约定不匹配：grader 调用 ars(f, c(-5,5), n=1000)，实现将第二参数按单个 lower 验证；任务文字未明示精确签名，存在规格歧义。",
    "bn-fit-modify": "学出的 DAG 边方向和目标不符，采样分布 KS 检验也失败。",
    "break-filter-js-from-html": "浏览器测试未出现预期 XSS alert；过滤绕过未成功。",
    "cancel-async-tasks": "3 tasks / max-concurrent 2 场景中启动两个任务，但 SIGINT 后没有预期的两个 Cleaned up 输出。",
    "chess-best-move": "输出 e5e6，期望 e2e4 与 g2g4。",
    "dna-assembly": "引物缺少 BsaI 位点前至少一个碱基的 clamp。",
    "dna-insert": "反向引物退火温度 55.857347，小于要求的 58。",
    "filter-js-from-html": "既有 XSS 向量未清除，也改动了应保持不变的 HTML。",
    "financial-document-processor": "发票分类与数量不正确，summary 12 行而要求 11 行。",
    "gcode-to-text": "输出 Embossed text，与应解出的文本不一致。",
    "model-extraction-relu-logits": "提取的矩阵全部 30 行未匹配目标。",
    "mteb-retrieve": "嵌入计算完成但检索排序结果 HumanEval 与期望 MTEB 不符；需再查模型调用/检索 prompt 与 pooling 约定，不能直接视为硬件故障。",
    "overfull-hbox": "修改后分词数 939 vs 原文 938，违反仅用 synonyms 替换的约束。",
    "polyglot-c-py": "除 main.py.c 外留下 cmain，违反唯一输出文件要求。",
    "protein-assembly": "融合蛋白片段排列未满足指定顺序。",
    "pytorch-model-cli": "预测 7，期望 2，多输入测试也未匹配。",
    "pytorch-model-recovery": "state_dict 多出 norm 权重，forward 只接受一个 Tensor 而测试需要两个。",
    "raman-fitting": "拟合峰中心/宽度/幅度/基线明显不符合数据目标。",
    "sam-cell-seg": "坐标写成 tuple，要求 flat list。",
    "sanitize-git-repo": "仍残留测试用 token，且有超出指定替换的改动。",
    "sqlite-with-gcov": "已经编译但只在交互 shell export PATH，独立 verifier 进程找不到 sqlite3；属于交付持久化问题，不能算成 harness 随机丢失文件。",
    "torch-tensor-parallelism": "RowParallelLinear 发生 mat1/mat2 维度不匹配 (2x4 vs 16x48)。",
    "video-processing": "起跳帧预测超出要求范围，含 1 vs [219,223] 的明显误差。",
}
OTHER_TIMEOUTS = {
    "caffe-cifar-10": "仍在修复 libcaffe/LMDB 链接，训练产物未完成；CPU 数更少可能影响编译，但日志不能证明配到 32 CPU 就会解出。",
    "circuit-fibsqrt": "仅 9 个已记录步骤，8 次输出截断，仍在构造电路；已完成调用的 token/时间不包含所有失败或被取消请求。",
    "feal-linear-cryptanalysis": "先有计算与终端恢复困难，后续输入/输出停滞；当前没有足够证据将其唯一归因到同一大段粘贴死锁。",
    "make-doom-for-mips": "仍在修复自建 MIPS 链接脚本，压缩后也未完成；未再出现旧的无限 fallback 失败循环。",
    "mteb-leaderboard": "下载/评分 subprocess 卡住且难以中断；网络与终端转发均可能参与，具体阻塞点待复现。",
    "nginx-request-logging": "已完成 API 调用累计 10435.998 秒，约 174 分钟；任务 cgroup 仅约 59 秒 CPU，另有 12/57 步解析错误。主要不是本地 CPU 持续不足。",
    "path-tracing": "已完成 API 调用累计 8071.265 秒，约 135 分钟；大量生成与协议错误，不能仅靠 CPU 升配解释。",
    "path-tracing-reverse": "已完成 API 调用累计 9342.034 秒，约 156 分钟；仍在逆向像素公式，结果未匹配。",
    "polyglot-rust-c": "仅 15 个已记录步骤却生成约 40 万 token，API 已记录约 107 分钟；仍在试错语法，cgroup CPU 约 61 秒。",
    "schemelike-metacircular-eval": "多次整份重写 evaluator，最终仍语法错误，grader 63/63 失败；GPU 生成/反复推理时间多于本地计算。",
    "train-fasttext": "cgroup 累计约 8.54 CPU 小时，资源差异值得验证；同时多次模型精度只到约 0.58，最后仍未达精度要求，不能预设升配后通过。",
    "winning-avg-corewars": "对部分对手胜率未达阈值；大部分时间用于模型生成与调整，并非本地 CPU 持续饱和。",
    "write-compressor": "压缩产物不能还原原文且大小不合格；长输出与压缩 fallback 后丢失进度可能拖慢，但旧无限重试 bug 未再出现。",
}
OTHER_EXECUTION = {
    "build-pov-ray": "模型把二进制/异常字符输出到终端后反复误判 shell 状态，最终 tmux session 消失。既有下载困难，也有 agent exit/控制符操作；不能仅因 RuntimeError 就认定外部环境故障。",
    "gpt2-codegolf": "异常位于本地 Singularity exec，而非模型 API；此前模型发出 pkill -9 python3，可能杀掉 Python 实现的环境控制服务。存在网络困难，但原任务模型权重已预置，不能把额外下载失败直接视为必需输入缺失。",
}


def main():
    rows = json.loads((OUT / "raw-evidence.json").read_text())["tasks"]
    assessed = []
    for row in rows:
        name = row["task"]
        if row["status"] == "passed":
            continue
        if name in ENVIRONMENT:
            cause, reason = ENVIRONMENT[name]
            group, confidence = "observed_environment_or_grader_blocker", "confirmed_obstacle_not_confirmed_recoverable_pass"
        elif name in RECORDING:
            group, confidence, cause = "recording_deadlock_consistent_with_reproduced_bug", "strongly_supported_task_linkage", "asciinema_2_2_blocking_io"
            reason = RECORDING[name] + "零模型调用的 35/70 KB 无害输入对照复现该死锁；直接终端正常，录制时卡死；1 KB 分块并节流的诊断实验完整写入。但尚未重新运行该题。"
        elif name in OUTPUT_FAILURES:
            group, confidence, cause = "graded_output_or_contract_mismatch", "grader_trace_observed_not_blanket_model_capability_claim", "solution_output_or_interface"
            reason = OUTPUT_FAILURES[name]
        elif name in OTHER_TIMEOUTS:
            group, confidence, cause = "other_timeouts_requiring_controlled_rerun", "mixed_or_unresolved", "timeout_mixed"
            reason = OTHER_TIMEOUTS[name]
        elif name in OTHER_EXECUTION:
            group, confidence, cause = "other_execution_errors", "mixed_or_unresolved", "execution_mixed"
            reason = OTHER_EXECUTION[name]
        else:
            raise RuntimeError(f"Unreviewed failed task: {name}")
        assessed.append({"task": name, "status": row["status"], "group": group, "cause": cause, "confidence": confidence, "assessment_zh": reason, "result_origin": row["origin"], "result_path": row["trial"] + "/result.json", "output_tokens_recorded": row["output_tokens"], "completed_api_seconds_recorded": row["api_seconds_recorded_completed_calls"]})
    assert len(assessed) == 55 and len({r["task"] for r in assessed}) == 55
    groups = Counter(r["group"] for r in assessed)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "model_calls_for_audit": 0,
        "official_score_percent": 51.5, "measured_score": {"passed": 34, "total": 89, "percent": 100*34/89},
        "gap_percentage_points": 51.5-100*34/89, "additional_passes_to_reach_or_exceed_51_5": 12,
        "groups": dict(groups), "observed_blocker_causes": dict(Counter(ENVIRONMENT[n][0] for n in ENVIRONMENT)),
        "conclusion_zh": "有确凿环境/判分问题，当前 38.2% 混入这些损失，不能直接当成模型能力结论。14 题确认遇到环境/判分障碍，另 3 题高度符合已独立复现的录制死锁；题量足以覆盖约 12 个额外通过所对应的分差，但无法从失败日志推断修复后会有多少题通过，尚未证明全部 13.3 分由环境造成。",
        "counterfactuals_not_rescored": {"all_9_infrastructure_errors_pass_hypothetically": 100*43/89, "all_14_observed_blockers_pass_hypothetically": 100*48/89, "note": "仅作题量上界，均不是修正后实测分数；未删除失败分母。"},
        "additional_protocol_findings": {
            "hardware": "官方 32 CPU/48 GB；本轮 4 CPU/32768 MiB。有确定 OOM 事件，CPU 贡献需要控制实验；许多超时主要花在模型调用/终端恢复。",
            "thinking_history": "89 题均未启用客户端 interleaved_thinking，未设置服务端 preserve_thinking；两端都要配置才能保留历史 reasoning。官方建议 agent 场景保留，但未披露 TB2 当次是否开启，影响未量化。",
            "composition": "当前为 73 个原始结果 + 16 个补跑结果，涉及多个服务端点与两个 agent 源码版本；官方为 5 次均值，当前只有一个组合 seed。端点成绩不能跨不同题目直接作因果比较。",
            "compression_bug": "此前 10 题已经补跑，不能把旧无限 fallback 循环再次当作当前 13.3 分的直接解释。",
            "sampling": "已核对的温度、top_p、top_k、3h、context 与 max_tokens 不是目前发现的主要协议差异。",
        },
        "recommended_validation_order": [
            "先做无需模型的机械修复验证：终端录制输入转发、网络隔离和低端口权限、Python 3.9 引导兼容、exec HTTP timeout 与 verifier budget 对齐、测试依赖/必需数据可直连取得。",
            "将 14 个有环境障碍的任务和 3 个录制死锁候选组成固定名单，统一主服务、32 CPU/48 GB、相同 agent 版本重跑；保留旧失败与新通过的完整轨迹，仅统计实测转化。",
            "thinking 保留作为独立变量，不能与所有环境修复混改后再宣称单一原因贡献。",
            "完整复现需全 89 题统一设置并做 5 次均值；当前 ASI/SWE 队列继续运行，调查没有改动或中止它。",
        ],
        "tasks": assessed,
    }
    files = ("raw-evidence.json", "terminal-recording-probe.json", "headless-network-probe.json")
    report["evidence_sha256"] = {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files}
    (OUT / "assessment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (OUT / "per-task-assessment.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assessed[0]))
        writer.writeheader()
        writer.writerows(assessed)
    (OUT / "environment-rerun-candidates.json").write_text(json.dumps({"not_yet_executed": True, "purpose": "fixed diagnostic task set, not an unbiased replacement for full benchmark mean", "tasks": sorted(ENVIRONMENT | RECORDING)}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"groups": dict(groups), "report": str(OUT / "assessment.json"), "candidate_count": len(ENVIRONMENT | RECORDING)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
