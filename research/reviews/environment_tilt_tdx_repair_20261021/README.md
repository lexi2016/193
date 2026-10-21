# Bounded original TDX diagnostics

These are byte-identical, independently reviewed source snapshots. They contain no market responses, runtime configuration, vendor source, or credentials. They are not an activated collector.

The original execution imported m33_tdx_source_refresh.py from the research module at revision 7f14e366e6c54e0c6dcdaeaac5a9924ecf6c9583. Provide that module on PYTHONPATH to run synthetic tests. The protocol test also requires the pinned tdxpy 0.2.7 source or installed package. Its original relative source location is recorded in the test; an installed package remains an import fallback. See RESULTS.json for the published sdist identity.

The tests are GBBQ_OFFICIAL_08/test_capture.py and TDXPY_PROTOCOL_09/test_protocol.py. Each passed two focused checks in the original reviewed environment. This publication does not claim an additional portable-environment test run. Actual executions require separately frozen configurations, activation receipts, registration, and the original bounded executor.

The GBBQ capture preserves raw events without assigning adjustment factors. The protocol diagnostic varies connection setup and header modes while preserving the original parser. Neither source review establishes full-market coverage, native adjustment factors, model performance, or production permission.
