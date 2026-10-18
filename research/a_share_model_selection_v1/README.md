# A-share volume-price model selection research

This source snapshot compares five nested feature groups (M0–M4) using daily Top10 rankings and observed-calendar close(T+h)/close(T)-1 for h = 1, 2, 5. All groups use the same finite-feature candidate keys for formal comparison. The frozen model is HistGradientBoostingRegressor; 90 formal fits and six engineering canary fits are planned.

At this snapshot, the M0 canary completed three actual fits. M4 is being retried after an independently reviewed shared-calendar lag boundary repair. Formal fits have not started and no model uplift is established. The repaired reporting path distinguishes missing labels from whole-date exclusions in paired comparisons.

`REVIEWED_SOURCE_STATUS.json` records file hashes, independent targeted verification, and execution counts. Synthetic unit tests are implementation checks, not evidence of investment performance. Real run artifacts and the append-only experiment registry remain local.

The export is an auditable source snapshot, not a standalone runnable training package: local frozen input bindings, original formula texts, native formula runtime, registry adapter, and market data are intentionally not published. The diagnostic utility also retains its original reviewed source binding; it is not a ready-to-run authorization for a new diagnostic on later model code.

Industry categories are retrospective; the market universe and calendar use the observed bound panel. Historical full-A eligibility and official-calendar parity are not certified. Results are capped at DIRECTION_ONLY. No costs, capital allocation, NAV, executable-alpha claim, production, BUY integration, or orders are included.
