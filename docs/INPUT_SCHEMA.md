# Portable frozen-factor predictor input

`tools/predict_frozen_factor.py` accepts:

* `--tpm`: CSV/TSV, optionally gzip; first column is human gene symbol, remaining headers are sample IDs. Values are finite, nonnegative **unlogged TPM**. Gene symbols and sample columns must be unique. Do not pass CPM, log expression or raw counts as if they were TPM.
* `--metadata`: CSV with unique `sample`, numeric `age`, and binary `male`/`sex_missing`. Required sample rows must match TPM columns exactly. Use the documented source encoding; `sex_missing=1` marks missing sex rather than treating missing information as a confirmed female.
* `--output`: a new local CSV filename; existing files are refused. Outputs sample ID, raw clipped factor prediction and optionally externally offset prediction. This is neutrophil percentage, not AD probability.
* `--model`: optional alternative NPZ path; default is the bundled protected source model.
* `--offset`: optional fixed intercept shift in percentage points, determined on a separate calibration set. The default is zero. No calibration is learned from the supplied evaluation rows.

Computation matches stage 54: log2(TPM+1), source gene scaling, fixed signed module weights, fixed feature scaling and ridge coefficients, clipping to [0,100]. Missing genes contribute zero on the source-standardized scale, retaining original denominators; available/missing counts are reported. No refitting or target-cohort gene rescaling occurs. The scorer refuses very low gene coverage (below 90% of the fixed model genes); this guard does not validate a new assay's clinical suitability.

For optional full point-metric replay, `tools/evaluate_saved_predictions.py --input-root PATH` expects the original snapshot `source_data` layout: measurement_anchor/source_OOF.csv and heldout_factor_predictions.csv, stage26/external_transport_predictions.csv, stage27/GSE249477_target_relative_predictions.csv and stage28/GSE248417_frozen_predictions.csv. Those participant-level files are deliberately not in this public snapshot. The tool does not download them, retrain models, regenerate intervals or certify clinical performance.
