# Execution order and reconstruction requirements

The original stage names are preserved under `workflow/`. Training scripts are scientific source archives, not an orchestrated clean-room workflow. Parameters, seeds and original hash guards are retained. The default portable audit and scoring commands are in `tools/` and do not call these training scripts.

## Blood main line

1. Obtain GSE63060/GSE63061 and platform annotation. Reconstruct the original gene-symbol mapping, expression matrices, formal labels and source-only scaling. The fixed 5,000-gene signed module definition is supplied in `workflow/modules/frozen_pca_modules.csv`; the fixed classifier is in `workflow/metadata/stage22/primary_classifier_parameters.json`. Initial R preprocessing and module-discovery scripts are not reconstructed in this snapshot.
2. Stage 22 contains nested validation and fold-local conformal sensitivity code. It needs `workflow/processed/`, module-score matrices and stage 20 relabelled records. Those intermediates are not bundled. Its signature-resource inputs also require the upstream MCPcounter resources; do not substitute random signatures.
3. Stages 26–28 contain independent array and RNA-seq scoring, platform availability and target-relative sensitivity code. Their processed source expression and official labels must be restored first. Target-relative RNA-seq mapping uses multiple target profiles and is transductive; it is not a single-patient deployed predictor.
4. For measured source prediction, obtain GSE157103 TPM/series matrix and the specified MSV000085703 SQLite release using `docs/data_sources.csv`. Stage 53 `run_measured_anchor_pilot.py` pairs exact RNA/clinical IDs and runs nested models, stress tests and within-background permutations. `freeze_and_verify_predictor.py` freezes an alpha-10 model; the already frozen NPZ is supplied. Do not rerun freezing into the supplied snapshot when only scoring is intended.
5. For external measured-endpoint evaluation, obtain the SoundLife metadata, unstimulated whole-blood H5AD and GRCh38.91 gene-length reference. Stage 54 `prepare_soundlife.py` reconstructs TPM and exact RNA/CBC matching; `run_external_validation.py` uses the frozen source model, 20 calibration participants and 74 held-out participants. Original protocols record exclusion/split rules. The 826 visit input is not 826 independent participants.
6. Stage 55 audits measurement-axis association and triage rejection; stage 56 supplies dimension-matched baselines. Stage 57 audits upstream resources and a possible new cohort; stage 58 implements the **official MCPcounter** comparison in R and two-group uncertainty. The new 138-person resource was not scored and is not an extra validated cohort.

## Brain supporting evidence

1. Stage 33 downloads the official SEA-AD multiregion Astrocyte/Immune pseudobulk H5AD releases and pairs RNA with measured pathology. Obtain the separate official donor/pathology metadata and brain-resource/Luminex files referenced by stages 32/33. These resources are not all resolved by the automatic downloader. Inspect the source filenames before execution.
2. Stage 35 contains helper model functions needed by stage 37; stage 37 derives donor-region quality/composition variables and controls.
3. Stage 39 defines the three-region pathology endpoints and nested donor-level modelling, permutations, region null controls and nuclear sensitivity. Stage 40 benchmarks alternative multi-view models and ablations. Source selection is performed within folds.
4. Stage 41 handles GSE160936 counts and within-group diagnostics. Obtain all 24 count archives and the family SOFT metadata; freeze source models through stage 39 before external expression processing. This branch evaluates RNA-component association in 12 donors, not external reconstruction of identical SEA-AD pathology units.
5. Stage 50 preserves exploratory paired protein and genetic qualification work linked to the blood triage framework. Failed batch transport is retained as a scope boundary; no MR/colocalization/virtual-knockout result was generated.

## Reconstruction matrix

| Branch | Supplied | Inputs still needed | Verified in this packaging round |
|---|---|---|---|
| Saved-result audit | Aggregate tables, hashes, audit entry | None | Integrity and 18 saved-value matches |
| Frozen neutrophil scoring | NPZ + portable input scorer | Locally obtained TPM and covariate CSV | Checked against existing study prediction arithmetic |
| Initial array workflow | Fixed modules/classifier and later validation source | Original preprocessing, mapped matrices, formal labels, stage 20 records | No clean-room reconstruction |
| Source/target training | Stage 53–58 scripts and protocols | Raw releases, intermediate records, original preservation-guard inputs | No retraining |
| Brain benchmark/transport | Stage 32–41 sources, protocols, aggregate results | Official donor/pathology metadata, paired pseudobulks, family SOFT, frozen brain models | No retraining |
| Figure exploration | Stage 62 calculation/plot source, aggregate gene tests | Corrected processed expression, saved module scores, paired RNA/CBC input | Source syntax checked; no figure regeneration |

## Environment and upstream implementation

Core numerical packages are pinned in `requirements.txt`. Historical stage 22 additionally needs `statsmodels` and `neuroCombat`; recorded versions for those two were not located, so `environment/requirements_optional_legacy.txt` deliberately does not invent pins. Official R scoring requires R and the pinned MCPcounter source/signatures from commit `b6eac73e91c246fcff0bb1a5c68a816cd588fc48` of `ebecht/MCPcounter`; stage 57 fetch code records the source URLs. Its downloaded third-party code/signatures are not vendored in this package. Original hash checks must pass; do not disable them to conceal mismatched inputs.

The downloader writes only requested inputs listed in the registry and checks historical checksums when available. Historical source links may change; no fresh network download was performed during packaging. Use `python tools/download_inputs.py --list` to inspect available downloads and byte sizes before selecting inputs.
