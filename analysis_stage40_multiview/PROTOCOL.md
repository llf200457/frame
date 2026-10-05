# Stage 40: locked multiview benchmark, 2026-10-01

This is continued exploration of previously used donors, not preregistration or an independent confirmatory experiment. Freeze this file before fitting. No changes to target, candidate list or parameter grid after inspecting outcomes.

## Cohort and targets
Reuse the stage39 76 complete donors and MTG Astrocyte/Micro-PVM inputs. Primary: Tau_panel (mean log1p AT8 in MTG/DFC/MEC). Auxiliary: Tau_DFC, Tau_MEC. One donor per row. Same five outer/three inner folds and seeds 2026/2027/2028. Stage39 genes ElasticNet is an existing reference; reproduce its predictions on a smoke fold. Every new model uses the same training-only 10-per-cell selection as that reference. No external cohort labels are used here. No best-seed reporting.

## Models and hypotheses
| Model | What it tests | Expected if component matters |
|---|---|---|
| genes_ridge | Sparse coefficients versus regularized dense linear fit, same 50 inputs | ElasticNet improves over Ridge if sparsity matters |
| random_forest | Strong nonlinear tree baseline, same selected genes/covariates | A candidate must beat this to support added modelling value |
| svr | Epsilon-insensitive nonlinear baseline, same inputs | Tests whether generic nonlinear regression explains improvement |
| joint_rbf | Ordinary RBF kernel on all 50 standardized inputs | Tests whether a custom multiview geometry is necessary |
| additive_kernel | Separate quality, Astro and Micro kernels, no interaction | Tests independent contributions of views |
| interaction_kernel | Additive kernel plus elementwise Astro*Micro kernel | Improvement over additive supports statistical interaction, not cell communication or causality |
| astro_kernel | Quality + Astro only, Micro removed | Micro contribution supported only if both-view prediction improves |
| micro_kernel | Quality + Micro only, Astro removed | Astro contribution supported only if both-view prediction improves |
| late_fusion | Mean of two separately fit quality+cell ElasticNets | Tests partitioned fitting versus pooled features, with no learned voting weight |

Kernel formula: K_C=(C C^T / p_C)/mean(diag(C C^T / p_C)); K_A=exp(-||A_i-A_j||^2/p_A), K_M similarly. K_add=K_C+(K_A+K_M)/2. K_int=K_add+K_A*K_M. Single-view K=K_C+K_view. Joint RBF exp(-||X_i-X_j||^2/p_X). All quantities fit only on the corresponding training split. Target is training-centered, standardized; intercept is training mean. Kernel solve (K+lambda I)^-1 y. Product is a positive semidefinite tensor-product kernel; it is an established mathematical construction, not by itself a new algorithm.

Grid (three choices/model): Ridge 10/100/1000; RF 128 trees, min_samples_leaf 2/5/10, max_features=1, seed=outer seed+fold; SVR C .1/1/10, epsilon=.1 standardized target, gamma=1/p; all kernels lambda .01/.1/1; late-fusion alpha .01/.1/1, l1_ratio=.5. Select by mean inner-validation MAE. No nonlinear high-dimensional gene input or deep learning in this small cohort. Raw denominator is all genes for CPM, unchanged from stage39.

## Judgement and uncertainty
Report MAE, R2, CCC for all seeds and targets. Paired bootstrap donors 4000 draws on per-donor mean errors across seeds; condition on fitted OOF predictions, not retrained bootstrap. Compare each model against stage39 genes and quality. Interaction-versus-additive is a prespecified component contrast. Descriptive percentile CIs are not adjusted; apply Holm to paired donor-level sign-flip tests across the full 9-model x 3-target genes-comparison family (27). All are exploratory because these donors have been repeatedly used. Advantages require consistent direction across seeds and CI excluding zero against strong baselines; independent validity remains a separate requirement.

## Independent resources
Audit original public sources before downloading. Record true donor count, endpoint and input compatibility. Braak stage is an ordinal proxy and cannot validate calibrated AT8 area prediction. Spatial spots/sections are nested in donors. Do not replace missing cell-specific inputs or quality fields silently, and do not tune frozen models using external outcomes. Access-restricted cohorts remain unavailable until authorized credentials/data are provided; this run does not request/sign data agreements.

## Outputs and compute
CPU only; 405 outer fits (9 models x 3 targets x 3 seeds x 5 folds), each with 9 inner fits; 4050 fitted predictors, two per fit for late fusion. Save all predictions, splits, selection, hyperparameters, convergence and hashes. Smoke-check prediction independence from test outcomes, PSD interaction matrix and reference reproduction before full run. Preserve failed experiments. Do not claim 405 independent experiments or 76 new donors.
