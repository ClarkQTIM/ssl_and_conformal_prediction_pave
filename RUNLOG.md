# RUNLOG: every command that produced evidence

Env for all runs: `PYENV_VERSION=dr_ssl_sl` (Python 3.12.0, `provenance/dr_ssl_sl_pip_freeze.txt`) unless noted.
Hashes are SHA-256 prefixes (16 hex). Data/outputs are not in git; this log is how they are identified.
Inputs: `analysis/six_model_seed_test2/*/3/lac/alpha_010/mc_lac_run_alpha0.1.csv` (full hashes in `ledger/claims_ledger.yaml`), `csvs/`.

| Run | What | Command | Code commit | Seed(s) | Result | Output hash |
|---|---|---|---|---|---|---|
| R-001 | Contract/audit tests | `python -m pytest tests -q -p no:cacheprovider` | 348ff52 | fixed in tests | 11 passed, 7 xfailed (F-002, F-009, F-010 evidence) | n/a |
| R-002 | F-002 split-sensitivity experiment | `python experiments/f002_split_sensitivity.py 200` | 77466c7 | `random_state` 0..199, image and patient schemes | gate passed: legacy split reproduces saved CPSL for 6 models exactly; outputs below | `f002_per_model.csv` 05411fcd41dba74b; `f002_table1_delong.csv` 358586fa2bc03bad; `f002_table4_ttests.csv` c9ae281be829493f |
| R-003 | Summaries for F-002/F-021/F-022 (incl. DeLong on all 1,686 images; overlap 237/309 women, 17.9% images) | `python experiments/f002_summarize.py > experiments/outputs/f002_summary.txt` | working tree after 77466c7 | n/a | see AUDIT.md F-021/F-022 | `f002_summary.txt` 44712339132bbea7 |
| R-004 | Paper Tables 1-4 + Sec. text recomputed from saved outputs; ledger generated | `python verification/reproduce_paper_tables.py` | working tree after 77466c7 | split seed 0 (as in saved outputs) | 108 ledger rows: 106 verified, 2 discrepancies (F-019) | `ledger/claims_ledger.yaml` 2ed64c8616 |
| R-005 | Static analysis | `ruff`, `vulture` from `audit_tools` env on `src/` | 7fa4a9d | n/a | reports in `provenance/static_reports/` | n/a |

## Earlier ad-hoc checks (before this log; superseded by scripts above)
- DeLong cross-check (`/tmp/delong_check.py`) -> replaced by `tests/test_crosscheck_stats.py` (synthetic) and R-004 (real).
- Table 2/3/4, N=481/669, accuracy-by-CPSL, calibration/test overlap: first computed inline, now reproduced in R-003/R-004.
- `/tmp` scripts are not part of the evidence chain.

## Not yet run (so nothing below may be claimed)
- **Phase 4: the legacy pipeline (`run_several_analyses.sh` -> `main.py` -> `analysis_new.py`) has NOT been re-executed.** All "reproduced" statements are from the saved legacy outputs.
- Figures (heatmaps, stacked GT, ROC) not yet checked against the paper.
- Upstream inference CSV provenance (models, checkpoints, split CSV) is Phase 8.
