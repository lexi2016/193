## 环境调制行业月份加分设计（未执行）

新增[设计草案](research/reviews/environment_tilt_design_20261020/README.md)：冻结M0后以单一观察市场广度调制原行业月份正先验，有界加分并学习一个历时λ。仅CONTRACT_DRAFT_NOT_EXECUTION_READY；全部五完整fold所需早期OOF元数据和原两年基础训练历史仍有缺口，未训练/预测/回测。原S1失败和G1 INCONCLUSIVE保留。
## 环境×行业×月份训练方案咨询材料（2026-10-20）

[阅读材料与可复制ChatGPT prompt](research/reviews/environment_training_advice_20261020/README.md)。补齐已完成的M0–M4环境实验、S1失败及G1/归因边界，供外部模型设计下一项有限实验；本批没有新训练或科学评价。

---

## G1 归因诊断补充（2026-10-20）

[归因报告](research/reviews/g1_attribution_20261020/REPORT.md) · [验收范围](research/reviews/g1_attribution_20261020/ACCEPTANCE.json) · [文件哈希](research/reviews/g1_attribution_20261020/EXPORT_MANIFEST.json)。冻结历史结果的描述性拆解及安全导出已独立复核；G1 信息门仍为 INCONCLUSIVE，S1 失败，S2 未启动。未训练、预测、重算区间或改变选股规则。

---

## 2026-10-20 研究审阅材料

[A股行业季节性 → S1 → G1 → q* 研究审阅包](research/reviews/g1_20261020/README.md)：源码、方法、聚合结果、负结果和独立审阅记录。S1开发门失败；G1最终区间跨零、信息门INCONCLUSIVE；S2未启动，整体DIRECTION_ONLY。未公开原始行情、逐日真值或模型权重，完整真实数据复现限制见包内说明。

以下保留仓库原有历史说明。

---

# 当前状态：R4 输入门补件待办（2026-10-18）

两份 JSON 的当前 raw 字节与既有捕获一致。修正后的缺失 SHA-256 常量复现器精确命中两条旧摘要（2/2），独立 Reviewer 已完成合成回归与收据复核。原 A 阶段的错误复现结论已由 A2 附记取代，历史工件保持不变。

已受限恢复原提供代码包的 19 个成员并完成 B2 静态报告。2022 年价格历史已有；现有 selector 增量机制不回填旧年份。原生 193 导出/版本身份，以及 INDEXC、历史名称/ST 等上下文输入的 PIT 绑定仍为 CANNOT_CONFIRM，训练门保持关闭，fit/backtest/prediction 均为 0。

独立验收：A2 合成测试 4/4、B2 合成测试 3/3；PASS 仅覆盖诊断工具修正、静态报告消费及对应收据生命周期，不代表研究突破或训练资格。

本地注册表：全局 route/event/execution/correction = 329/2489/680/85；V193 = 9/20/11/0，未终止执行 0。全局未注册路线及 contract-only 数量未在本范围审计，均为 CANNOT_CONFIRM。此次公开同步只包含必要源码、测试和安全摘要，原注册表与私有捕获件没有上传。

源码依赖见 B2 的 SOURCE_ALLOWLIST.json；其中既有 registry adapter 与 execution guard 是本地运行依赖，本次未复制。B1 模块供 B2 复用静态构建器，其原报告入口已有失败记录；最终修正入口为 B2。本文下方及旧报告保留历史状态，以本段和 R4 安全摘要为当前结论。

RESEARCH_ONLY=true；production_allowed=false；buy_gate_updated=false；orders_sent=false。

---

# V193 R2 / R3 review entry

The current no-fit result and annual-boundary interpretation are in the [R3 progress report](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3/public_export/PROGRESS_REPORT_R3.md) and [R3 safe aggregate](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3/public_export/SAFE_AGGREGATE_R3.json). These report the strict `trade_date` audit and annual coverage gaps without linking to private tables or the batch evidence index. The [transport summary](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3/public_export/CURRENT_SOURCE_TRANSPORT_SAFE_SUMMARY_R3.json), [focused test output](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3/public_export/CALENDAR_TEST_RESULTS.txt), and [R3 dependencies](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3/public_export/DEPENDENCIES_R3.md) describe the source-capture evidence and review setup.

Public R3 source and regression coverage: [date audit](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3/public_export/materialized_inputs_r2/targeted_acceptance_r3/audit_panel_trade_dates_r3.py), [strict calendar parser](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3/public_export/materialized_inputs_r2/targeted_acceptance_r3/calendar_dates.py), [test](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3/public_export/materialized_inputs_r2/targeted_acceptance_r3/test_calendar_dates.py), and its [legacy R2 parser dependency](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3/public_export/materialized_inputs_r2/run_materialization_r2.py).

R2 context: [initial materialization report](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/public_export/PROGRESS_REPORT.md), [R2 safe aggregate](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/public_export/SAFE_AGGREGATE.json), [R2 materialization runner](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/public_export/run_materialization_r2.py), [R2 continuation helper](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/public_export/recover_missing_inputs_r2.py), [R2 dependencies](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/public_export/DEPENDENCIES.md), and [continuation dependencies](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/public_export/CONTINUATION_DEPENDENCIES.md). Latest captured-source evidence: [capture report](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/continuation_capture_r1/public_export/CURRENT_SOURCE_CAPTURE_REPORT.md), [capture safe summary](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/continuation_capture_r1/public_export/CURRENT_SOURCE_CAPTURE_SAFE_SUMMARY.json), [capture helper](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/continuation_capture_r1/public_export/capture_r2_payloads.py), and [capture dependencies](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/continuation_capture_r1/public_export/CAPTURE_DEPENDENCIES.md).

Native V193 identity and trainable coverage remain `CANNOT_CONFIRM`. R3 is an engineering-only date-token canary and does not establish model or training acceptance.

# 192 / 193 选股模型研究进度

研究证据截止：2026-10-18 06:09 UTC（北京时间14:09）；其后归档和发布说明属于文档管理记录，不计新增实验。

这是研究进度摘要，非交易系统、投资建议或收益保证。

当前研究状态：**PARTIAL**。尚未证明稳定改善或胜过原193；多年新滚动轨为 **NOT_RUN**。

- [中文进度报告](PROGRESS_REPORT.md)
- [机器可读状态](status.json)

截至2026-10-18北京时间14:09整理：已完成历史探索和行业短窗试验。报告列出不同批次的结果、负结果、失败、未运行项和数据缺口；不同支持集、窗口和指标不合并排名。

本次报告编制新增训练0、推断0、模型审计0。数据恢复与连接复核不计模型成果。报告内百分比为相应支持集的持有期标签均值，非账户净收益。

本仓库公开同步研究进度摘要；研究证据截止时间和结论不因发布而改变。

本摘要版仅提供研究进度，不包含私人证据链接、逐股预测、原始行情、模型权重或部署能力。历史指标仅按报告注明的证据范围使用，不代表可复现的完整数据集已经公开。

