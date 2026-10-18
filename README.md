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
