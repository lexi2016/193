# G1 saved-result attribution audit

Read REPORT.md and ACCEPTANCE.json together. This is a descriptive DIRECTION_ONLY audit of existing G1, with the scientific information gate still INCONCLUSIVE. No fit, prediction, model load, new CI, re-ranking or strategy was run.

The CSV files contain frozen aggregate contribution/validity tables only. Real stock/date-level rows, predictions, labels, raw inputs, models and local paths are excluded. Source code and synthetic tests are audit references; machine-bound inputs and registry dependencies are intentionally absent, so this is not a standalone research runtime or a deployment package.

One portability adaptation replaces the original registry ROOT literal with the required FACTOR_FACTORY_ROOT environment variable; the numerical core, runner and synthetic tests are copied exactly. PROVENANCE.json records original and published source hashes. This adaptation adds no algorithm or permissions. Acceptance is limited to the frozen audit and safe-export scope.
