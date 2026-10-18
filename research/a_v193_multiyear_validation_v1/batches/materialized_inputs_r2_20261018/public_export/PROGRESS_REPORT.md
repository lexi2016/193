# V193 R2 materialization progress

This export records a bounded no-fit materialization and a physical date/security-key projection over frozen 20261015 panels.

- Byte-identical materialized source documents: 3 of 5.
- The V193 snapshot text has the same SHA-256 as the V192 CAND3 comparison text.
- Native V193 producer identity remains CANNOT_CONFIRM.
- Annual panel rows report physical row, date, and security coverage only. V193 trainable coverage remains CANNOT_CONFIRM because feature parity, PIT eligibility, complete warmup, and label gates were not evaluated.
- The only panel columns read by this batch were trade_date and ts_code.
- Fit calls: 0; backtests: 0; predictions: 0; labels read: 0; weights read: 0.
- Research-only flags: research_only=true, production_allowed=false, buy_gate_updated=false, orders_sent=false.

The public aggregate contains no Drive IDs, account-local source paths, formula text, raw market rows, labels, predictions, weights, or run logs.

The included runner source depends on the project's existing OfficialRegistryAdapter and ExecutionGuard; this export is not a standalone runtime package.
