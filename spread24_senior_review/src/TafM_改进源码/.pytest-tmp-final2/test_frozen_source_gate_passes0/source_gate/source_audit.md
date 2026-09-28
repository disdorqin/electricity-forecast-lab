# Source Gate Audit

- STATUS: **PASS**
- Frozen base SHA256: `a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea`
- Base rows: `40670`
- Complete target through: `2026-08-21`
- Partial tail: `2026-08-22` hours `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]`
- Merged source rows: `40670`
- Merged source SHA256: `72f5b2dd9571536abbd52a64d666d4d80351953012cd96ff2c4d326a94d5f32d`
- Extensions: `0`
- Dependency evidence: `VERIFIED_CANONICAL`
- Schema hash verification: `VERIFIED`
- Blocking gaps: `[]`
- Vintage evidence: `{'CONTRACT_INHERITED': 259}`
- Quarantine starts: `n/a`

## Errors

- None

## Boundary assertions

- target = DA - RT
- forecast origin = D-1 14:00
- complete labels <= D-2
- target-day actual / DA and D-1 h15-h24 realized are not candidate features
- frozen base file is read-only; extensions merge through immutable append/correction sources
