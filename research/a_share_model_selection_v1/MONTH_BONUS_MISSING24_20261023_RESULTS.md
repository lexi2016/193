# 月份行业加分：24 个缺失日期定向核对（2026-10-23）

实际诊断和独立实际验收已完成：`PASS_INDEPENDENT_MISSING24_ACTUAL_FLAGS_REGISTRY_AND_REUSE`。实际验收 SHA `ad35218d6c8ccf12da0e93a6bafb712562d7fc53b5887b95ded8121a6a843747`；[审查终态收据](/Users/lexi/Desktop/学习/因子工厂/research/a_share_model_selection_v1/batches/environment_tilt_execution_20261020/MONTH_BONUS_MISSING_24_16/ACTUAL_REVIEW_CLOSEOUT.json) SHA `9aa4c2989b041edc23c708999aaadf99b2a6fe91fa6c8f59c38df38ba79a50ea`。新源代码、配置、来源门禁和26项新测试已获独立PASS。

24个缺失日期对应26个股票键。四个已冻结来源的精确T+2目标行均不存在：原canonical、原QFQ raw、backtest保留层、selector保留层。原canonical/raw及backtest的26个信号行均有效；selector有22个信号行，其余4个也不存在。没有发现原raw有行而canonical丢行的投影矛盾。

当前可恢复日期为0，恢复delta为空。本次无法恢复的直接证据是四源精确目标均无行；口径桥缺失只约束潜在替代值的准入。停牌与采集缺失无法由这些字段区分，保持`STATUS_UNKNOWN`，不补0、不用nextbar或forward-fill。保留层价格口径桥未证明，其signal精确价格parity和float32/float64差异已单列诊断，未用于label。

| 信号日 | 股票代码 | 精确T+2目标日 | 四源目标分类 |
|---|---|---|---|
| 2024-01-12 | 603955.SH | 2024-01-16 | 均无行 |
| 2024-03-26 | 002951.SZ | 2024-03-28 | 均无行 |
| 2024-04-26 | 603559.SH | 2024-04-30 | 均无行 |
| 2024-04-26 | 603959.SH | 2024-04-30 | 均无行 |
| 2024-08-27 | 600777.SH | 2024-08-29 | 均无行 |
| 2024-10-24 | 000796.SZ | 2024-10-28 | 均无行 |
| 2024-11-14 | 000908.SZ | 2024-11-18 | 均无行 |
| 2025-01-22 | 688350.SH | 2025-01-24 | 均无行 |
| 2025-03-10 | 000905.SZ | 2025-03-12 | 均无行 |
| 2025-04-21 | 300561.SZ | 2025-04-23 | 均无行 |
| 2025-04-29 | 300225.SZ | 2025-05-06 | 均无行 |
| 2025-04-30 | 300225.SZ | 2025-05-07 | 均无行 |
| 2025-06-04 | 002762.SZ | 2025-06-06 | 均无行 |
| 2025-06-25 | 603389.SH | 2025-06-27 | 均无行 |
| 2025-07-10 | 603719.SH | 2025-07-14 | 均无行 |
| 2025-08-05 | 603389.SH | 2025-08-07 | 均无行 |
| 2025-09-02 | 605255.SH | 2025-09-04 | 均无行 |
| 2025-09-03 | 605255.SH | 2025-09-05 | 均无行 |
| 2025-09-11 | 002968.SZ | 2025-09-15 | 均无行 |
| 2025-09-29 | 603843.SH | 2025-10-09 | 均无行 |
| 2025-10-28 | 603843.SH | 2025-10-30 | 均无行 |
| 2025-11-12 | 000609.SZ | 2025-11-14 | 均无行 |
| 2025-11-12 | 002713.SZ | 2025-11-14 | 均无行 |
| 2025-11-13 | 002713.SZ | 2025-11-17 | 均无行 |
| 2025-11-17 | 000592.SZ | 2025-11-19 | 均无行 |
| 2025-12-04 | 600800.SH | 2025-12-08 | 均无行 |

原461个完整配对日直接复用，没有重算、改值或覆盖；逻辑SHA为 `4ffa0756a1467fed34e7491e911f3e7dd932e839e709ac22916367f816d1812a`。旧ledger文件SHA为 `ca6d727953a66467ec331180deff15fff71b343785acb0ace49c33a3bda43f59`。

因此本次诊断没有改变旧稳定性结果：主样本485个名义日、461个完整配对日，24个缺失日期保持；C−A描述均值为−0.001843个百分点，2/4半年为正。连续有效区段仍含短于20日者，冻结MBB20/B2000/seed20261021的区间上下限继续为null。`screen=false`、`CANNOT_CONFIRM_STABILITY_FROZEN_BLOCK_INTERVAL`、`DIRECTION_ONLY`。这不确认完整485日样本的真实收益方向。旧aux/partial/2026H1全部引用，未重跑。

实际诊断耗时 25.291 秒，采样峰值RSS 1319370752 bytes；4线程/4GiB/600秒上限。本次官方登记1路线、6事件、1执行、0correction（4个flags-only checkpoint及START/TERMINAL）。没有fit、predict、网络、大盘过滤或生产/交易操作。

实际交接：`/Users/lexi/Desktop/学习/因子工厂/research/a_share_model_selection_v1/batches/environment_tilt_execution_20261020/MONTH_BONUS_MISSING_24_16/DIAGNOSTIC_ACTUAL_HANDOFF.json`，SHA `1d97adcec2dc3eb85d513c561b7e0f57bba82279e5d8af481bc8f4db6ff36d13`。
恢复影响摘要：`/Users/lexi/Desktop/学习/因子工厂/research/a_share_model_selection_v1/batches/environment_tilt_execution_20261020/MONTH_BONUS_MISSING_24_16/RECOVERY_IMPACT_SUMMARY.json`，SHA `0cab040d31a4bdb850976c1fef7e285f2fc16e31e2255af8fb0c847d373f91f4`。

`RESEARCH_ONLY=true`、`production_allowed=false`、`buy_gate_updated=false`、`orders_sent=false`。
