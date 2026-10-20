# SFT训练

核心代码只有四份：

- `train_mace_sci.sh`：独立 Bash 入口，训练参数显式写在命令中。
- `train_agent.sh`：Agent 独立 Bash 入口，用 `positive` / `full` 选择两版数据。
- `prepare_data.py`：按数据集制作、校验训练 JSONL。
- `runtime.py`：缓存准备、磁盘保护和最终 HF 权重校验。

当前版本使用 **Science + Math**：Qwen3.6-35B-A3B，2 节点 × 8 卡，64K，packing，global batch 32，1 epoch。学习率 **1e-5 → 1e-6**，cosine，warmup 3%。不划分验证集。

Agent 两版都从原始 **Qwen3.6-35B-A3B** 独立开始：16 卡，**256K、packing、TP/SP 16、EP 16、Expert TP 1、CP 1**，micro batch 1 / global batch 8，1 epoch，学习率 **1e-5 → 1e-6**，cosine，warmup 3%，完整重计算和全部优化器 CPU offload。

| Agent 版本 | 输入数据 | 长度过滤前条数 |
|---|---|---:|
| `positive` | `rts` + `rlmath` + `naturebench` | **12,729** |
| `full` | `mindforge` + `rts_all` + `rlmath_all` + `naturebench_all` | **68,567** |

先用同一组 16 卡运行 `positive`，导出完成后运行 `full`；两个版本不会接续对方的权重。完整集包含未成功轨迹，并额外包含没有逐条 reward 的 MindForge；此比较同时改变数据量、成功率及教师组成。两版均为 1 epoch，更新步数随数据量变化。最终实际保留条数和 tokens 写入各自的 `cache_stats.json`。

每 100 步保存完整训练状态，全部保留。共享盘余量低于安全储备时，在保存完整 checkpoint 后停止。最终导出的 HF 权重经过分片、参数键名/形状和全部数值有限性检查后移入 `ckpts`；这项检查不代表评测质量最优。

```bash
# 在双节点任务中运行；平台提供节点编号与 MASTER_ADDR：
bash /mnt/data/rui.yan/projects/coding_agent/sft/train_mace_sci.sh \
  /mnt/data/rui.yan/projects/coding_agent/outputs/r01_运行编号

bash /mnt/data/rui.yan/projects/coding_agent/sft/train_agent.sh positive \
  /mnt/data/rui.yan/projects/coding_agent/outputs/r02_运行编号_a
bash /mnt/data/rui.yan/projects/coding_agent/sft/train_agent.sh full \
  /mnt/data/rui.yan/projects/coding_agent/outputs/r02_运行编号_b
```

| 内容 | 位置 |
|---|---|
| 原始快照 | `../data/training/` |
| 处理好的训练 JSONL | `data/`，按数据集和筛选版本分文件，全是训练集 |
| 中间 checkpoint、运行日志、代码快照 | `../outputs/<运行编号>/` |
| 最终 HF 权重 | `../ckpts/<运行编号>_hf/` |
| token 缓存、临时空间 | `../cache/`；优先使用 Worker 本地临时盘 |
| 统计、SHA256、逐条来源 | `../logs/sft/` |

数据保留完整推理、动作与观察；超出窗口的样本整条删除。SCI 两台 Worker 分别编码 Science 和 Math；Agent 按数据集交错分配编码任务。缓存完成后一起训练。模型、源数据和运行环境均在挂载盘上，离线运行，不自动上传。

## 数据

| 文件 | 条数 | 内容 |
|---|---:|---|
| `math.train.jsonl` | 545,431 | DeepSeek-V4-Pro High，数学 CoT/TIR |
| `science.train.jsonl` | 294,306 | DeepSeek-V4-Pro，推理档位未标注 |
| `pdbthink.train.jsonl` | 24,522 | GLM-5.3，蛋白结构推理 |
| `proofs.train.jsonl` | 51,018 | V4-Pro Max 证明/修改；仅教师自评通过，未独立验证 |
| `proofs_critics.train.jsonl` | 288,376 | V4-Pro Max 证明评审，单独使用 |
| `mindforge.train.jsonl` | 1,001 | GLM-5.2，保留上游 SFT 轨迹及结束动作 |
| `rlmath.train.jsonl` | 1,441 | Kimi-K3，reward=1、无超时且明确完成 |
| `naturebench.train.jsonl` | 38 | K3/GLM-5.3，评分有效、正常结束、纯文本且工具往返完整 |
| `rts.train.jsonl` | 11,250 | GLM-5.3，3,316 个任务；保留同题的不同完整成功轨迹，仅去除内容完全相同的样本 |
| `rts_all.train.jsonl` | 66,012 | GLM-5.3，完整轨迹，包含未通过成功筛选的结果 |
| `rlmath_all.train.jsonl` | 1,445 | Kimi-K3，完整轨迹，不限制任务奖励为 1 |
| `naturebench_all.train.jsonl` | 109 | K3/GLM-5.3，完整文本轨迹，不限制实验指标改善 |

以上均为训练时长度过滤前数量。原始 Agent 精选四份共 13,730 条；严格成功三份共 12,729 条，完整四份共 68,567 条。所有版本均按同一 JSONL schema 整理。RTS 不按任务去重；同题多条轨迹并不增加任务覆盖。

基座原生模板统计：MindForge 全量约 2.03 亿输入 tokens，平均约 202K；64K 内只有 6 条，256K 内 973 条。这里包含被遮罩的用户/工具观察，不是监督 tokens。Agent 训练需单独配置长度，不能直接套用当前 SCI 的 64K 整条删除策略。完整长度统计见 `../logs/sft/agent_token_stats.json`。MindForge 的 messages-only 源文件不带实例 ID，不能按共享题面推断独立任务数。

RTS/RLMath 的 Terminus 批命令映射为 `terminal_batch` 工具，保留命令顺序、等待时长、终端输出及完成标记；原任务文本保留，原 JSON 协议说明替换为对应工具说明。NatureBench 的工具名称和调用参数来自真实记录，参数 schema 根据观测到的参数类型归纳；不补造未记录的思考或工具输出。MiMo 没有现成教师回答，未转换成直接 SFT 样本。

当前校验覆盖格式、内容哈希和来源，不代表所有样本都经过本地重新执行，也尚未完成与四个评测的全面重叠排查。

```bash
# 增量处理指定数据集（源文件保持不动）：
/mnt/data/rui.yan/venvs/OmniRL/bin/python prepare_data.py \
  --datasets mindforge rts rlmath naturebench --resume

# 额外生成完整集版本，保留上述成功集文件：
/mnt/data/rui.yan/venvs/OmniRL/bin/python prepare_data.py \
  --datasets rts rlmath naturebench --agent-all --resume

# 全量校验 JSONL、行数、SHA256 和来源记录：
/mnt/data/rui.yan/venvs/OmniRL/bin/python prepare_data.py --check
```
