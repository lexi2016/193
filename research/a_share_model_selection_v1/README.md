# A-share model-selection research results

The frozen M0–M4 experiment has completed all 90 formal fits (five model arms, three target horizons, six rolling half-year folds) plus the original six engineering canary fits. All 90 formal checkpoint hashes match their metadata and recorded replay receipts; recorded predictions and Top10 parity are exact. An independent audit reconciled the actual result CSVs with the machine metrics.

M1–M4 have negative same-date paired mean return differences versus M0 at T+1, T+2 and T+5. M4 fails the frozen composite acceptance gate at all three horizons. M1–M3 comparisons are exploratory. Exact metric definitions, rounded aggregate tables, sample counts and limitations are in `FORMAL_RESULT_SUMMARY.json`.

These are gross close-to-close outcomes on exposed history, not tradable net returns or sealed-forward evidence. Industry classification is retrospective. The evidence ceiling remains `DIRECTION_ONLY`. No model promotion, BUY change or order is authorized.

The separate single-cell 12-thread engineering test made one fit and returned `PARITY_REJECTED`. Further diagnosis and testing are paused at the user's request. Its reviewed startup code is retained for traceability; it is not an accepted performance replacement.

Inputs, private formula texts, native runtime, local bindings, model weights, row-level predictions and the full registry are excluded. This source snapshot is not a standalone reproducible data package. Source identities and review scope are in `REVIEWED_SOURCE_STATUS.json`.
