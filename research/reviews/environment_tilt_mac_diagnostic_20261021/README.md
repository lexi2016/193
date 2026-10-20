# Mac source-semantics diagnostics

This package preserves five independently reviewed Python source/test files byte for byte and summarizes two bounded local diagnostics in [RESULTS.json](RESULTS.json).

The scalar reconstruction did not establish numeric parity for any of six fixed symbols. The subsequent frozen affine/cent hypothesis matched all 240 held-out OHLC values, with zero unrounded residual in this sample. All calibrated slopes were 1. Five windows contained no cached action; only one contained one event. Volume/amount differences remain separate.

These are **OPENBOOK input-semantics diagnostics**. Native QFQ and rounding conventions, action completeness, PIT and training readiness remain unconfirmed. Actual-result independent interpretation was pending when this package was frozen. Matching this sample does not authorize prefix generation or training.

The modules depend on non-shipped local producer sources, private runtime/config/input bindings and an existing official registry adapter. This is not a standalone data pipeline. No market rows, models, credentials, local receipts, abandoned Windows adapters or private execution configuration are included.

Reviewed source files:

- [Local recovery](../../a_share_model_selection_v1/environment_tilt_mac_recovery.py)
- [Registry lifecycle](../../a_share_model_selection_v1/environment_tilt_mac_registry.py)
- [Affine diagnostic](../../a_share_model_selection_v1/environment_tilt_mac_affine.py)
- [Recovery tests](../../a_share_model_selection_v1/tests/test_environment_tilt_mac_recovery.py)
- [Affine tests](../../a_share_model_selection_v1/tests/test_environment_tilt_mac_affine.py)

[SHA256SUMS](SHA256SUMS) uses repository-root-relative paths. Run its verification from the repository root. Research only; production, BUY gate and orders remain disabled.

The exact reviewed recovery test retains one explicitly forbidden negative-fixture path constant. It is not a real input binding or credential; source sanitation would change the reviewed bytes.
