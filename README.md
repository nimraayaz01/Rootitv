# itvcheck — diagnostics for machine-learning studies on compiled trait-variability data

Accompanies the manuscript *What can a compiled root-trait dataset support? Grouped validation, effective sample size and a
calibrated null for intraspecific trait variation*. Pure NumPy / SciPy / scikit-learn (no statsmodels).

| Module / script | Purpose |
|---|---|
| `itvcheck.fit_reml`, `lr_test` | Crossed-random-effects linear mixed model (REML/ML), optional known measurement variance; validated against the statsmodels fits of the earlier analysis |
| `itvcheck.cv_sampling_var`, `cv_noise_var_n`, `bias_correct_cv` | Sampling variance of a CV (plug-in vs outcome-independent), small-sample bias correction |
| `itvcheck.LegacyGroupKFold` | Pins the historical (scikit-learn <= 1.5) GroupKFold assignment. **scikit-learn >= 1.6 breaks ties differently, which changes fold membership and the CV results.** |
| `itvcheck.grouped_oof`, `skill_table` | Nested site-grouped CV (optional precision weights), pooled skill against the training mean with a site-cluster bootstrap |
| `itvcheck.fast_en_cv`, `fold_cache` | Fast elastic-net nested CV used for permutation nulls and design simulations |
| `groot_overlap_grid.py` | Sensitivity of an overlap-exclusion rule to its thresholds (needs the harmonized GRooT file) |
| `scripts/01..08_*.py` | The analyses in the paper, in order; each writes CSVs to the working folder |
| `PREREGISTRATION_external_tests.md` | Template for freezing external-test rules before data are opened |
| `tests/test_validate.py` | Regression test against the published v5 numbers |

Usage: put `Source_data.xlsx`, `Combined_SRL_ITV_Han_Weemstra_v2.xlsx`, `Weemstra_ITV.csv` in a folder, `export ITV_DATA=that/folder`,
run `python tests/test_validate.py`, then the scripts in order. Runtime on one core: about 1.5 hours in total
(the random-forest leave-one-site-out and repeated-CV runs dominate).
