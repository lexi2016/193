# 月份×行业固定加5分扩样结果（2026-10-23）

状态：`CANNOT_CONFIRM_STABILITY_FROZEN_BLOCK_INTERVAL`，`DIRECTION_ONLY`。独立源码、13项受影响测试及实际派生结果读回已通过（`PASS_INDEPENDENT_ACTUAL_DERIVED_RESULTS_AND_REGISTRY`）；研究结论保持不变。实际验收 SHA `ba62f4f57f2a78f25b4120880e9a088f04aebc0a9f703009fb9d58bd95e6dafc`；审查终态收据 [`ACTUAL_REVIEW_CLOSEOUT.json`](/Users/lexi/Desktop/学习/因子工厂/research/a_share_model_selection_v1/batches/environment_tilt_execution_20261020/MONTH_BONUS_STABILITY_15/EXPANSION_FILTER_RECOVERY_04/ACTUAL_REVIEW_CLOSEOUT.json) SHA `76e1e4f8669f4b837fab3d6cee4aea35d777166b93dcad7c01a4a6bc7cf86921`。

读取修复后本次扩样已正常完成。2023H2 已保存选择直接复用，2026H1 复用旧结果单列；没有重训、重新推理、网络采集或大盘过滤。

| 样本 | 名义/完整配对日 | A均值 | C均值 | C−A（百分点） |
|---|---:|---:|---:|---:|
| 2024–2025 主样本 | 485/461 | 3.162272% | 3.160429% | -0.001843 |
| 2024H1 | 117/114 | 3.133069% | 3.142155% | +0.009086 |
| 2024H2 | 125/122 | 4.554566% | 4.452273% | -0.102292 |
| 2025H1 | 117/110 | 2.717632% | 2.816055% | +0.098423 |
| 2025H2 | 126/115 | 2.139489% | 2.137467% | -0.002022 |
| 2023H2（不同输入版本辅助） | 124/122 | 3.291221% | 3.426583% | +0.135362 |
| 2026H1（旧结果） | 116/108 | 1.312470% | 1.347579% | +0.035109 |
| 2026H2 至10/15（partial） | 71/69 | 1.050258% | 1.143639% | +0.093382 |

主样本2/4半年均值为正；冻结要求至少3/4。主样本24个交易日因所选股票精确T+2标签不完整而整对剔除，未补0、换nextbar或补票。连续有效区段存在短于20日者，按冻结合同95%区间上下限为null，screen=false。有效配对子集均值是描述统计，完整名义样本均值为null。

A/C 最差5%日篮子尾部均值为 -6.113771% / -6.061591%。这是重叠T+2毛收益篮子比较；无NAV、成本后收益或投资组合回撤结论。

本次运行 227.502 秒，采样峰值RSS 1765113856 bytes；4线程/4GiB/600秒上限。官方registry本次1路线/8事件/1执行/0 correction（6 checkpoint，START及TERMINAL），无未闭合本次执行。全局routes/events/executions/corrections=434/4598/1424/122；全局未登记/contract-only未审计，外部持久化CANNOT_CONFIRM。

实际交接：`/Users/lexi/Desktop/学习/因子工厂/research/a_share_model_selection_v1/batches/environment_tilt_execution_20261020/MONTH_BONUS_STABILITY_15/EXPANSION_FILTER_RECOVERY_04/ACTUAL_HANDOFF.json`，SHA `b76cd42e9b7e5a13ef9939fccab99057a106ea93a4aa0545a3535e2d5973f7fc`。
机器摘要：`/Users/lexi/Desktop/学习/因子工厂/research/a_share_model_selection_v1/batches/environment_tilt_execution_20261020/MONTH_BONUS_STABILITY_15/EXPANSION_FILTER_RECOVERY_04/COMPACT_RESULTS.json`，SHA `39807c9a850ef3532e9c6f4ed302e744ffb20a959ea0e2f7a1329211eda895f5`。

`RESEARCH_ONLY=true`、`production_allowed=false`、`buy_gate_updated=false`、`orders_sent=false`。
