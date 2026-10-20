# A股行业季节性 → S1 → G1 → q* 研究审阅包

截至 2026-10-20，结论为 **DIRECTION_ONLY**：S1 开发门失败，G1 的正残差点估计尚未获得信息门通过，S2 未启动。这里公开已完成研究的算法、聚合结果、负结果及独立审阅记录，供方法和证据审查。

| 阶段 | 已有结果 | 结论边界 |
|---|---|---|
| 2006–2025 行业季节性 | 89 行业；严格稳健季节增量仅 1 个行业×月份格（C40，11月） | 当前行业分类回溯、包内同日 Top100 描述；不能当预测收益 |
| S1：M0 加 4 列行业月份先验 | h2 Top10 配对增量 −13.05394 bp；原 block 区间 [−34.96726, −5.39352] bp；正 fold 1/5，五门皆失败 | `S1_DEVELOPMENT_GATE_FAILED`；564/601 完整配对日 |
| 原 G1 信息诊断 | 高低先验组 h2−M0预测残差差 +29.75649 bp；正 fold 5/5 | 原 bootstrap 与等有效日估计的权重不一致，旧正区间未获接受 |
| G1 CI 权重修复 | 187/547 有效日没有合法 20 日 block 覆盖 | 正确停止，CI=null，真实 bootstrap=0 |
| Bartlett20 HAC，常数 1.95996 | MA5、异方差 MA5 的校准未通过 | 正确停止，未执行真实区间 |
| 八情形 q* | q*=2.9877439760955102；区间 **[−6.01749, +65.53048] bp**，547/601 有效日 | `INCONCLUSIVE`；八种指定 DGP 校准/验证通过不等于真实数据普适 95% 覆盖 |

G1 是 M0 Top50 内的条件信息诊断，5 bp 是资源筛选线。G1 不是 S1 的 Top10 策略增益，也没有证明未来赢家 Recall@K、可执行净收益或封存集突破。20 年描述窗口与 2024H1–2026H1 五个完整模型评价 fold 分开；2026H2 部分期只展示诊断。

- [方法、公式和冻结门槛](METHOD.md)
- [依赖、复现限制及验证范围](REPRODUCIBILITY.md)
- [聚合结果与历史失败链](evidence/aggregate_results.json)
- [历史独立审阅收据投影](evidence/review_receipts.json)
- [冻结配置](evidence/frozen_configs.json) · [20年中文解释报告](evidence/seasonality_report_zh.md)
- [源码来源及原件 SHA-256](PROVENANCE.json) · [依赖清单](DEPENDENCIES.json) · [发布字节校验](SHA256SUMS)

`src/` 保留统计核心和原有小型合成测试；`archive/` 保存编排、历史描述与 M0 依赖源码，供阅读。路径脱敏和 qstar 的相邻 HAC 文件导入调整均在 provenance 逐项声明。个人绝对路径替换为逻辑引用或不可用占位符，原件字节保留在本地。JSON 收据投影不是原件签名，也不表示本次发布包已经通过审阅。

真实逐日行情、标签、预测、有效日 mask、模型权重和本机运行依赖未公开。**本包不能独立重现真实 G1 区间、原 mask 下的 480,000 次模拟或 S1 训练。** 合成测试中的 mask 是测试夹具，不能替代原研究 mask。公开 SHA 便于持有原件者核对，不证明未公开数据的真实性、PIT 或许可。

安全状态：`research_only=true`、`production_allowed=false`、`buy_gate_updated=false`、`orders_sent=false`。不包含投资或生产晋升结论。
