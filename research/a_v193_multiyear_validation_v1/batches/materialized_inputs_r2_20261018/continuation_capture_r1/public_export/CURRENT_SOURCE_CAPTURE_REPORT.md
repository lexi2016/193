# V193 R2 current source byte capture

This bounded no-fit capture stores the current raw JSON payloads under filenames containing each payload's actual SHA-256. It preserves the bytes without normalizing them and separately compares current hashes with the historical R2 expected hashes. A matching current capture does not retroactively certify the older V193 source identity.

- `input_data_manifest`: 4026 bytes, actual SHA-256 `b02813c24916488c03684513bac36da6ed39bf87a8ae919f8b5f2f6cc5a7e516`, historical SHA-256 `ef60a6c9141c8d1dd871c6815fc250b2be171f1f535d6ebfcf016d40ddb6b8cd`, match=false, JSON parse=PASS; saved at `continuation_capture_r1/private/captures/input_data_manifest__sha256_b02813c24916488c03684513bac36da6ed39bf87a8ae919f8b5f2f6cc5a7e516.json`.
- `independent_verification_receipt`: 1296 bytes, actual SHA-256 `e81c184e96e18f39c9884c1c448071b8c73a5b560b4600e0af00aea0df6766f1`, historical SHA-256 `7ec5e16286a0291484c7a80ebf8b24cea36479cb1448d24dfb548742587f7ae4`, match=false, JSON parse=PASS; saved at `continuation_capture_r1/private/captures/independent_verification_receipt__sha256_e81c184e96e18f39c9884c1c448071b8c73a5b560b4600e0af00aea0df6766f1.json`.

Raw capture/readback status: PASS. Historical identity: CANNOT_CONFIRM. Fit calls, backtests, predictions, label reads, and weight reads: all 0.
Research-only flags: research_only=true, production_allowed=false, buy_gate_updated=false, orders_sent=false.
