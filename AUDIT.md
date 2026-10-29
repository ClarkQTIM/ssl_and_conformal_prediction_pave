# AUDIT LOG

Rules: findings are logged, not fixed. Each gets a verdict from Chris before any change.
Severity: **S1** changes a paper number / invalidates a claim; **S2** wrong but number unaffected, or unverified risk; **S3** hygiene.
Status: `suspected` -> `confirmed` (by test/recomputation) -> `verdict` (fix / re-run / document) -> `resolved`.

| ID | Sev | Status | File:line | Description | Affects paper number? | Evidence | Verdict |
|---|---|---|---|---|---|---|---|
| F-001 | S2 | confirmed | many (48+ refs in `src/`, `aa_*/src/`, `bash_scripts/`) | Hardcoded absolute paths use `/scratch90/chris/...` (dir is now `chris_`) and `/data/SEED` (missing). Code cannot run unmodified. | No (blocks reproduction) | `ls /scratch90/chris` -> not found; path scan of baseline commit | Phase 4: minimal documented path patch; Phase 6: config-driven paths |

## Findings detail
(Add one section per finding: reproduction steps, expected vs actual, test that demonstrates it.)
