# 固定版本的官方评测代码

本目录保留运行和复核评测所需的官方代码、许可证与 Git 版本信息；数据集中保存在同一套评测目录的 `datasets/`。

| 目录 | 保留用途 |
|---|---|
| `swe-bench-pro/` | V2 locked-protocol Agent、补丁重评分、探针、HARD-51 ID 与发布校验记录 |
| `hle/` | 官方模型调用示例、答案判分器和评测协议 |
| `asi-bench/` | 官方 CLI、Agent 适配器、OS 沙盒、任务运行配置和评分器 |

SWE 采用 Git sparse-checkout，仅检出 V2 评测核心。旧版 V1 的 Dockerfile、运行脚本、历史轨迹、分析文件、网页，以及额外的 642 题工作副本已移除。正式使用的 HARD-51 位于 `../datasets/swe-bench-pro-v2-hard51/`，全部保留。

上游 README 中的 `v2/tasks` 属于完整上游仓库布局；本地运行时请使用上述独立 HARD-51 数据目录。各仓库准确 commit 见 `../archive/benchmarks.lock.json`。ASI 与 HLE 的运行、评分代码没有裁剪。
