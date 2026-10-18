# R3 review dependencies

The date parser uses only Python's standard library. The focused test uses `unittest`, the R3 parser, and the original R2 materialization runner solely to reproduce the legacy parser behavior. The panel audit used PyArrow 25.0.1 with a Python 3.12 runtime.

To rerun the focused parser regression from this export, run `python -m unittest -v test_calendar_dates.py` with the working directory set to `materialized_inputs_r2/targeted_acceptance_r3`. The neighboring `materialized_inputs_r2/run_materialization_r2.py` is included as the test's import dependency.

`audit_panel_trade_dates_r3.py` is included for source review. Its operational contract, input binding, panel paths, and registry lifecycle receipts are private batch evidence and are intentionally not part of this public export; do not treat the copied checker as a standalone rerun package.
