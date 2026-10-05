# Environment scope

Python 3.13.11 and all eight packages in the root requirements.txt were inspected in the working analysis environment on 2026-10-05. `requirements_stage26_recorded.txt` and `recorded_brain_environment.json` are untouched historical records. `statsmodels` and `neuroCombat` are not present in that inspected environment; optional legacy requirements explicitly remain unpinned. R packages and platform-specific BLAS/font behavior are not locked by the Python requirements.

No dependency installation, new model training, or complete fresh-environment replay was performed for this package. The portable integrity audit and downloader use only the standard library. The fixed predictor uses NumPy/pandas; its arithmetic replay against existing study predictions is tested separately in validation_report.json.
