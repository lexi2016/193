# Environment tilt OOF canary export

This packet exports four frozen research source/test pairs whose bounded independent code reviews passed. It also records the M23 time-use preflight, M24 single-cell fit identity check, and M25 controlled resource-guard stop. The packet is a provenance handoff; it does not establish alpha, full OOF performance, official-calendar certification, or production readiness.

M23 preflight used 485 nominal dates (480 nonempty plus 5 previously proven empty), with 124 score dates / 637,523 full-pool rows and 478 training dates / 2,057,675 rows. The maximum training-label end date was 2023-06-30, before the first score date 2023-07-03. The independent time-use review passed its narrow scope. First-source PIT certification and official-calendar status remain CANNOT_CONFIRM.

M24 performed one frozen fit and produced 637,523 score rows; the recorded roundtrip was exact and elapsed time was 33.238783 seconds. Recorded peak RSS was 9,304,473,600 bytes (about 8.67 GiB). The independent model review passed single-cell fit-receipt and score-identity checks. No performance evaluation or full OOF alpha result is established; first-source PIT and official calendar remain CANNOT_CONFIRM.

M25 stopped at OWNER_RSS_LIMIT under a 2 GiB parent cap before any worker or query was dispatched. Actual worker count, query count, prefix/comparison emissions, and fit count were zero. Exact parent peak RSS is null because it was not recorded; no OS OOM was observed; no retry or cap change occurred.

Evidence SHA-256: M23 command `1a8e82165caf192604b08ca894e7c8d665b422a09d74e2a0786c0f1b99b626fb`; M23 independent review `24e31ddb708935da59776600e414336e7288dff92f5441a715f5d20fa515e547`; M24 command `12138088823a7a4e17c04a21ed4633065d268dcddf8310d691b547aeeea6aedb`; M24 independent review `49c19b98cbd326971d0b023ad493dd45bd91939f4f5aabb6246dab4d1f7c1011`; M25 command `0a482e966acf4e8e5cd8aa5736587242f80b7de54fec2ab948b9aa64e186cfc4`; M25 failure evidence `a4c2cc4be485c6a2a71f640ef0548df2b7ce8d1f99c6c235143e8febcbc234c8`; M25 timing `95ad46e235595ce15be9434e6ae53ca562005e66ad1b06026202950a55ea4917`.

Research state remains DIRECTION_ONLY; no breakthrough, BUY-gate, production, or order claim is made. See `RESULTS.json` for the machine-readable summary and `SHA256SUMS` for exported file checksums.
