# Environment-tilt recovery progress packet

This packet contains eight byte-identical source/test snapshots for the independently reviewed M11_R1, M12, M13_R1, and M14 components, plus compact progress evidence for M15 and M16. The exact review scopes and identities are in `RESULTS.json`; `SHA256SUMS` covers the eight snapshots and two public documents.

The independent reviews passed within their bounded software scopes. M11_R1 covers raw-response preservation, M12 covers synthetic model-core behavior, M13_R1 covers the resource-probe repair and direct regressions, and M14 covers pure source composition with synthetic fixtures. These reviews do not establish actual data readiness, source/PIT certification, official history or calendar parity, downstream consumer integration, or model acceptance.

M15 is a one-symbol, one non-unit event candidate: 5,036 of 5,036 OHLC comparisons matched to cents across 1,259 overlap dates; the result remains `DIRECTION_ONLY` and does not establish full-market or training readiness. M16 failed before cache access because `psutil` was missing; the later M13_R1 resource repair passed eight independent tests, but the actual preflight was not rerun.

No included receipt establishes an actual training or OOF fit. M17, M18, M19, and M20 are outside this packet scope. Research safety remains `RESEARCH_ONLY`; production and BUY gates are unchanged, and no orders were sent.
