# Measurement-anchored blood transcriptomic triage audit

Research code and aggregate results for **Measurement-anchored auditing of blood transcriptomic triage in Alzheimer's disease**. This is a curated upload snapshot of the existing analysis, not a new experiment, a validated diagnostic service, or a complete raw-data training image.

## Quick start

```console
python tools/audit_release.py
python tools/download_inputs.py --list
python tools/predict_frozen_factor.py --help
```

The first two commands require only Python's standard library and do not download anything. The audit checks file integrity, protected model definitions and 18 saved numerical correspondences (point estimates and interval endpoints). **It compares saved aggregate numbers; it does not recompute patient-level metrics or bootstrap intervals.**

For the runnable fixed predictor, install the core dependencies in an isolated Python 3.13 environment:

```console
python -m pip install -r requirements.txt
python tools/predict_frozen_factor.py --tpm input_tpm.tsv.gz --metadata input_metadata.csv --output local_outputs/predictions.csv
```

Input TPM must be nonnegative, unlogged, human gene-symbol-level, genes by samples. Metadata contains `sample,age,male,sex_missing`. No example patient record or synthetic performance is substituted for the study data. See [the input schema](docs/INPUT_SCHEMA.md). An external intercept offset is optional and must be determined using a separate calibration set; the default is the unadapted frozen source predictor.

## Contents and execution scope

| Location | Contents | Scope |
|---|---|---|
| `tools/` | Integrity audit, frozen-model scoring, input downloader, optional saved-prediction evaluator | Portable command-line entry points |
| `workflow/` | Original scientific analysis scripts, protocols and fixed definitions | Historical source; raw and some intermediate inputs must be reconstructed |
| `results/` | Original aggregate tables, gene-level differences, selection/benchmark summaries | Saved results; no participant-level clinical or prediction records |
| `docs/` | Data sources, workflow order, schemas, claim boundaries and copy provenance | Records exactly what is and is not supplied |
| `environment/` | Recorded environment and optional historical dependencies | Core installed versions checked; full clean-install/retraining not tested |

Start with [WORKFLOW.md](docs/WORKFLOW.md) before running historical scripts. Their sibling directory names intentionally match the original project. Some intermediate inputs and original preprocessing steps are not supplied; downloading all listed files is **not** sufficient to guarantee end-to-end reproduction. The table in WORKFLOW states these gaps. Never run all files in alphabetical order.

## Evidence interpretation

The strongest measured-endpoint result is external prediction of neutrophil percentage after an intercept calibration using 20 participants, evaluated in 74 different participants (R² 0.466; MAE 4.743 percentage points). The two-group bootstrap R² interval is [0.177, 0.608]. This is not independently established AD diagnosis. Superiority over the official MCPcounter baseline and an improvement in AD triage utility are not established. Brain analyses are supporting tissue-context evidence, not blood-to-brain causal validation. See [claim boundaries](docs/CLAIM_BOUNDARIES.md).

No GitHub URL, release DOI, authorship details or open-source license grant has been invented. See [LICENSE_NOTE.md](LICENSE_NOTE.md) and [citation information](docs/CITATION.md). Upstream raw datasets and third-party implementations must be obtained under their own terms. This snapshot excludes manuscripts, journal correspondence, access tokens, local environments and downloaded papers.
