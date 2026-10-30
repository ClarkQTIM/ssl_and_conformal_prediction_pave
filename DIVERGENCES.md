# DIVERGENCES: every difference between the legacy code/results and anything we change, and why

Rule: nothing in `src/`, `aa_*/`, `bash_scripts/` changes without a row here. Each row states what changed, why,
the finding that motivated it, the Chris verdict, and the measured effect on every affected paper number
(before -> after, same seed, tolerance stated). The paper is edited only to match a row marked `applied`.

## Code state
- **2026-10-29: `src/` is byte-identical to the original** (`diff -rq src /scratch90/chris_/ssl_conformal_pave/src` is empty).
  All audit work so far is additive: `tests/`, `experiments/`, `verification/`, `ledger/`, docs. No legacy behaviour has been altered.
- Original untouched: `/scratch90/chris_/ssl_conformal_pave` (manifest `provenance/original_manifest.sha256`, local only).

## Divergence table
| ID | Finding | File:function | Legacy behaviour | New behaviour | Why | Verdict | Effect on paper numbers (before -> after) | Status | Commit |
|---|---|---|---|---|---|---|---|---|---|
| (none yet) | | | | | | | | | |

## Planned / candidate divergences (NOT applied; each needs a verdict first)
| Finding | Candidate change | Expected effect | Needs verdict |
|---|---|---|---|
| F-001 | Config-driven paths (Phase 4: minimal patch only, to run legacy; Phase 6: proper config) | None on numbers | Yes (Phase 4 patch) |
| F-002/F-021 | Patient-level (STUDY+ID) calibration split and/or report spread over many splits; DeLong on all images | Changes Table 4 means/stars; Table 1 DN121-Severe star would go (p=0.087 on all 1,686 images) | Yes: pending |
| F-009 | Align `soft_prediction` with its image | None on paper (column unused) | Yes |
| F-010 | Textbook k-th order statistic in `find_quantile` | Slightly smaller sets (about 1/(n+1) coverage); changes CPSL numbers marginally | Yes |
| F-011 | Replace bare `except:` | None expected | Yes |
| F-014 | Report both uncorrected and Bonferroni-corrected stars (Chris verdict 2026-10-29); define one family size | Adds a column of stars; no number changes | Decided in principle; family definition pending |
| F-019 | Fix text 97% -> 96% and 30% -> 31% (l.432) | Two text numbers | Yes |
| F-003, F-016 | Fix 71,976 vs 72,967; validation 6% vs 5.4% | Text numbers | Yes |

## Rule for fidelity checks in Phase 6
Refactored code must reproduce `ledger/claims_ledger.yaml` (`reproduced_refactor`, `abs_diff`) with the legacy seeds. A
difference is allowed only if it traces to a row above; otherwise it is a regression.
