# Audit, Refactor & Provenance Plan: SSL + Conformal Prediction (PAVE)

Goal: be able to defend every claim, number, hyperparameter, and model in the paper, with a traceable chain
from the paper -> script -> config -> input CSV -> model/checkpoint, and reproduce the results with clean code.

Scope: this repo starts at the **inference CSVs** (model outputs). Training/inference code and the inherited
patient-split CSV live elsewhere and are audited in a follow-up (see Phase 8), but their *outputs* are
checked here against the split.

## Ground rules

1. **Original is never modified.** `/scratch90/chris_/ssl_conformal_pave` is the untouched source. All work happens here.
2. **Audit logs, it does not fix.** Findings go in `AUDIT.md`. Fixes happen later, one at a time, after a verdict from Chris.
3. **Evidence over opinion.** Reading code is a hypothesis. A finding is "confirmed" only when a test, a recomputation, or a run shows it.
4. **Every function earns its name.** If a function is called `add_one_plus_one`, it must return 2. Every function used in a paper result gets a contract test with hand-computed expected values.
5. **No data in git, ever (assume patient-level info).** `.gitignore` is an allowlist (`.py`, `.sh`, `.md`, `.yaml` only). Data/outputs are described by a SHA-256 manifest, not committed. Notebooks are excluded until outputs are stripped.
6. **Pushes are done by Chris.** Claude provides the command; Claude never has the token.
7. **Two independent passes** on every file that feeds a paper number (second pass by a fresh-context reviewer).
8. **Every run is logged:** command, git commit, seed, environment, input hashes, output hashes.

## Phases

| # | Phase | Output | Exit criterion |
|---|---|---|---|
| 0 | Snapshot | Hashed original, verified copy, git baseline, this plan | Copy verified == original manifest (**done**) |
| 1 | Inventory | `INVENTORY.md`: entry points, call graph, data flow, config/hparam list, dead-code candidates, which code produced which paper artifact | Every paper table/figure maps to a script (or is flagged "unknown") |
| 2 | Code-only audit | `AUDIT.md` findings; contract tests for core functions | All files read twice; checklist below complete |
| 3 | Paper-in-context audit | Paper claims extracted into the ledger; code re-read against methods/equations | Every method statement matches code, or a finding is filed |
| 4 | Legacy reproduction | Path-patched legacy run vs. paper numbers | Each ledger row has `reproduced_legacy` + diff |
| 5 | Triage | Verdict per finding (fix / re-run / document) | Chris signs off on every finding |
| 6 | Refactor | Clean package, dead code removed, tests | Refactored outputs == Phase 4 outputs (same seeds, tolerance stated) |
| 7 | Walkthrough + final reproduction | `paper_walkthrough.md` generated from the ledger; full re-run | Every ledger row verified / discrepancy explained |
| 8 | Upstream (follow-up) | Same treatment for training, inference, and split creation | Separate plan |

Phases 4 and 5 may reorder or loop: a bug found in Phase 2 can change what "correct" means in Phase 4.
Bugs that change a paper number are fixed and the paper updated (paper edits are still possible).

## The audit checklist (Phase 2 and 3)

### Statistical / conformal correctness
- [ ] **Quantile correction:** threshold uses `ceil((n+1)(1-alpha))/n` (finite-sample) and the right order statistic (`k`-th smallest, 1-indexed vs 0-indexed). Check `n` is the calibration-set size actually used.
- [ ] **Nonconformity scores:** LAC / APS / RAPS etc. match their textbook definitions (including randomization/tie-breaking and `<` vs `<=` on the threshold).
- [ ] **Coverage and set size:** coverage = mean(true label in set); set size = |set|; computed on the test set only, with the right axis/denominator. Empty sets handled and reported.
- [ ] **Exchangeability:** how calibration and test are drawn; no reuse of calibration points in test; no tuning of alpha or thresholds on test labels.
- [ ] **Alpha handling:** alpha values map to the coverage targets reported (e.g. `alpha_010` -> 0.10 -> 90%).
- [ ] **MC-dropout aggregation:** how the `mc_epistemic_*` columns are averaged/combined; sets built from the mean vs per-sample; the number of MC samples is what the paper says.
- [ ] **Metrics vs. definitions:** ROC-AUC, sensitivity/specificity, confusion matrices, correlation, DeLong, t-tests, bootstrap CIs, each cross-checked against an independent implementation (sklearn / scipy / numpy / a conformal library).

### Leakage, splits, and unit of analysis
- [ ] **Patient-level splits:** no patient appears in more than one of train / val / calibration / test. Check on the correct key (`STUDY` + `ID`, not `ID` alone; IDs may collide across studies). Includes MATCH_SET (matched case-control sets split across partitions?) and multiple images per patient.
- [ ] **Unit of analysis:** metrics, conformal calibration, bootstrap, DeLong, and t-tests operate at the right level. If many images per patient, resampling and independence assumptions must be at patient level, not image level.
- [ ] **Inherited split CSV audited:** uniqueness, class balance per split, overlap, duplicates, and consistency with the `dataset` column in each inference CSV.
- [ ] **Row alignment:** predictions are joined to ground truth by key, not by row position; no silent reordering after merges/sorts/filters.
- [ ] **Subsetting/filtering** (e.g. removal by floor average, endpoint removal): the filter uses no test labels in a way that leaks, and the post-filter n is reported.
- [ ] **Normalization statistics** computed on the correct split (and what norms were applied at inference).
- [ ] **Augmentations** applied at test time (and train vs eval mode, BatchNorm/dropout state, in the inference outputs we consume).

### General code correctness
- [ ] **Seeds:** every RNG (numpy, python, torch, sklearn, bootstrap) is seeded, seeded where intended, and the seed is recorded; seed-dependent results are reported with the seed set used.
- [ ] **Off-by-one / indexing:** class index <-> label mapping (binary vs three-class), `argmax` vs set membership, slicing bounds, `range` ends, 0/1-based thresholds.
- [ ] **Silent failures:** bare `except`, `errors='ignore'`, `fillna`, `dropna`, `astype` truncation, integer division, in-place mutation of shared DataFrames, chained assignment.
- [ ] **Parsing:** prediction columns stored as stringified lists (`"[0.1, 0.9]"`), so check `eval`/`literal_eval`, length, sum-to-1, and NaNs.
- [ ] **Hardcoded values:** thresholds, alphas, paths, class counts, model lists. Anything that should be a parameter but is a literal.
- [ ] **Dead/duplicated code:** `analysis_functions.py` vs `analysis_new.py` (note: `analysis_new.py` **is live**. It writes the paper's `*_eval_setlength_heatmap.png` and `*_stacked_gt_dist.png`, lines 805 and 872, into `analysis_outputs_eval/`), `dev_script.py`, `troubleshooting.py`, duplicated `cuq_classification_algorithms.py`, and `create_cuq_csv.py` (top-level vs `aa_*`).
- [ ] **Naming vs behavior:** each function's name, docstring, and behavior agree (contract tests).

## Tooling (non-LLM checks carry the weight)
- Static: `ruff`, `pyflakes`, `mypy` (loose), `vulture` (dead code), `bandit` not needed.
- Dynamic: `pytest` contract tests with hand-computed answers; **synthetic-data validation** (exchangeable data -> empirical coverage must land at about 1-alpha over many trials; known-AUC data -> AUC matches); property tests (coverage monotone in alpha; set size monotone in 1-alpha).
- Cross-implementation: key quantities recomputed independently (sklearn, scipy, numpy, optionally MAPIE/crepes).
- LLM review: fresh-context subagent passes per file; `/simplify` during refactor; `/review` on the refactor diffs.
- Run logging: a wrapper/hook records command, commit, seed, env, and input/output hashes for each run.

## Provenance ledger (Phase 3 -> 7)
`ledger/claims_ledger.yaml`, one entry per claim / table cell / figure panel:
`id, paper_location, claim_text, reported_value, script, function, config/hparams, alpha, seed(s), input_csv (+sha256), model/checkpoint, split, reproduced_legacy, reproduced_refactor, abs_diff, status (verified | discrepancy | unreproducible | unchecked), notes`.
`paper_walkthrough.md` is **generated** from the ledger so the two cannot drift.

## Git & push protocol
- Remote: `https://github.com/ClarkQTIM/ssl_and_conformal_prediction_pave.git` (private for now; public later).
- Commits are local; Claude gives Chris the `git push` command; Chris runs it.
- Before the repo goes public: re-scan history for data/IDs/paths; confirm no CSV/PNG/PPTX/IPYNB ever entered history.
- Planned commit sequence: baseline (done) -> docs -> inventory -> tests -> audit fixes (one commit per finding) -> refactor.

## Phase 0 status (done)
- Original hashed: 3,089 files -> `provenance/original_manifest.sha256`; copy verified identical.
- Git initialized (branch `main`); baseline commit contains **only** 38 `.py`/`.sh` files + `.gitignore`. Scanned: no secrets, emails, or long ID literals in code.
- **First finding (F-001):** all hardcoded paths use `/scratch90/chris/...` (renamed to `chris_`); `/data/SEED` also missing. Code cannot run as-is. See `AUDIT.md`.

## Open questions
1. Is top-level `src/` definitely what produced the paper? (`aa_Comparison_SL_SSL_PAVE/` is believed to be older; to be confirmed in Phase 1 by mapping paper artifacts to scripts.)
2. Location of the paper source / PDF (needed for Phase 3).
3. Location of the inherited patient-split CSV and the original images (`/data/SEED` is gone).
4. Which conformal library, if any, was used vs. hand-rolled (`cuq_classification_algorithms.py` appears hand-rolled).

## Where everything is documented (kept current)
| What | File |
|---|---|
| Findings (bugs, paper/claim problems), severity, evidence, verdicts | `AUDIT.md` |
| Every claim/table cell -> script -> input hash -> split/seed -> reproduced value (GENERATED) | `ledger/claims_ledger.yaml` via `verification/reproduce_paper_tables.py` |
| Every difference between legacy code and any change, why, effect on paper numbers | `DIVERGENCES.md` |
| Every command run as evidence: command, commit, seeds, env, output hashes | `RUNLOG.md` |
| Executable evidence for findings | `tests/` (xfail(strict) = confirmed bug) |
| Experiments that quantify a finding | `experiments/` (outputs untracked, hashed in `RUNLOG.md`) |
| Environment | `provenance/` |
| Plan, ground rules, checklist | `PLAN.md` |
| Code-to-paper map (Phase 1) | `INVENTORY.md`: **NOT YET WRITTEN** |
| Generated narrative for reviewers (Phase 7) | `paper_walkthrough.md`: **NOT YET GENERATED** |
