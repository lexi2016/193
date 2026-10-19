# A-share observed daily Top100 environment statistics

This research-only route ranks the same-day observed QFQ close-to-close pool by return, using the adjacent previous shared-calendar close and positive current-day volume. It keeps `UNKNOWN` industry observations in all denominators, uses a deterministic return-descending/code-ascending Top100, and reports descriptive industry lift, month, lagged CSI 300 regime, coverage, ties, and yearly parity. It does not build labels, fit a model, backtest a strategy, update a BUY gate, or send orders.

For lagged market regime, T-1 comes from the bound shared calendar. The reference is the nearest 20 bound index rows strictly before T-1; all 20 closes must be finite and positive. Missing history remains `UNKNOWN`, and the target-day index close is never used.

The frozen input binding points to the existing shared panel and its bound calendar, retrospective CAPCO industry member, and Futu-recovered CSI 300 series. The route reuses the bound `model_selection.py` readers and official locked registry adapter; all external dependency paths and SHA-256 values are recorded in the batch binding. The run uses the existing CPU runtime recorded in `CONFIG.json`.

Run synthetic checks with:

```sh
PYTHONPATH=research/a_share_top100_environment_v1 \
  /Users/lexi/Documents/Codex/2026-08-13/factor-mining-mac/.venv-training-v1/bin/python \
  -m unittest discover -s research/a_share_top100_environment_v1/tests -v
```

The original R1 full-statistics execution is terminal `FAILED_NO_FIT`; its identity and seven output tables are retained. Do not rerun it. The narrow boundary correction is a separate derived-artifact batch with its own frozen contract, binding, config, and identity. The first B launch failed before `RUN_STARTED` because string source-snapshot paths were treated as `Path` objects; the failure and unchanged pre-repair source are retained in `batches/market_boundary_repair_r1/PREFLIGHT_01_PROVENANCE.json` and `source_snapshots/`. The corrected attempt uses a new identity and execution ID. Once those R2 files are frozen, run this command once; the `run` command appends `RUN_STARTED` before reading the bound aggregate tables or index:

```sh
PYTHONPATH=/Users/lexi/Desktop/学习/因子工厂/workspace/.task_worktrees/a-top100-environment-20261019/research/a_share_top100_environment_v1 \
  /Users/lexi/Documents/Codex/2026-08-13/factor-mining-mac/.venv-training-v1/bin/python \
  /Users/lexi/Desktop/学习/因子工厂/workspace/.task_worktrees/a-top100-environment-20261019/research/a_share_top100_environment_v1/market_boundary_repair.py run
```

After a successful repair terminal, register the independent boundary review before the reviewer reads the repaired tables:

```sh
PYTHONPATH=/Users/lexi/Desktop/学习/因子工厂/workspace/.task_worktrees/a-top100-environment-20261019/research/a_share_top100_environment_v1 \
  /Users/lexi/Documents/Codex/2026-08-13/factor-mining-mac/.venv-training-v1/bin/python \
  /Users/lexi/Desktop/学习/因子工厂/workspace/.task_worktrees/a-top100-environment-20261019/research/a_share_top100_environment_v1/market_boundary_repair.py review-start
```

The parent coordinates the read-only reviewer against the registered source, delta, receipt, and seven repaired tables. The parent/writer terminalizes that review only after receiving its handoff, using `review-terminal --receipt <handoff.json>`; the reviewer does not write to the registry.

Outputs for the corrected attempt remain under the dated batch `outputs_attempt_02/` directory. Results are `DIRECTION_ONLY`; the exposed historical pool, retrospective industry classification, 2021/2026 partial-year coverage, and index provenance limits are stated in `COMPACT_REPORT.md` and the machine receipts.
