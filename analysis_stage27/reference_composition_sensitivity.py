"""Label-free RNA-seq reference-composition sensitivity for the locked score.

All 42 AD/control endpoint profiles are scored each time. The 42 profiles
used to estimate target gene means/SDs are randomly selected from all 62
without consulting diagnoses. No model coefficient is fitted or selected.
"""
from pathlib import Path
import csv
import gzip
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE if (BASE / 'modules').is_dir() else BASE.parents[1]
OUT = ROOT / 'analysis_stage27'
modules = pd.read_csv(ROOT / 'modules/frozen_pca_modules.csv')
parameters = json.loads((ROOT / 'overleaf_upload_route2_v7/source_data/stage22/primary_classifier_parameters.json').read_text())
meta = pd.read_csv(OUT / 'GSE249477_linked_metadata.csv').set_index('sample_id')
matrix = ROOT / 'raw/GSE249477/GSE249477_raw_count_normalize_04-10-2025.csv.gz'
header = next(csv.reader(gzip.open(matrix, 'rt', encoding='utf-8', errors='replace')))
tpm_cols = [c for c in header if c.endswith(' - TPM')]
raw = pd.read_csv(matrix, usecols=['Name'] + tpm_cols, compression='gzip', low_memory=False).dropna(subset=['Name'])
named = raw.Name.astype(str).str.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
complete = raw[tpm_cols].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
raw = raw.loc[named & complete]
assert raw.Name.is_unique and len(tpm_cols) == 62
expr = np.log2(raw.set_index('Name')[tpm_cols].astype(float) + 1)
expr.columns = [c.split(' (GE)')[0] for c in tpm_cols]
assert set(expr.columns) == set(meta.index)

genes = modules.gene.tolist()
shared = [gene for gene in genes if gene in expr.index]
assert len(shared) == 4686
expr = expr.reindex(genes).to_numpy(dtype=float)
measured = np.isfinite(expr[:, 0])
assert int(measured.sum()) == 4686
samples = [c for c in meta.index if meta.loc[c, 'status'] in ('AD', 'CTL')]
col_index = {name: i for i, name in enumerate([c.split(' (GE)')[0] for c in tpm_cols])}
endpoint_idx = np.array([col_index[name] for name in samples])
y = meta.loc[samples, 'status'].eq('AD').to_numpy(dtype=int)
assert len(samples) == 42 and y.sum() == 21

features = parameters['features']
groups = []
for feature in features:
    group = modules[modules.module.eq(feature)]
    groups.append((np.array([genes.index(g) for g in group.gene]),
                   np.sign(group.loading_signed.to_numpy()), len(group)))

def score(reference_idx):
    reference = expr[:, reference_idx]
    means = np.zeros(len(genes), dtype=float)
    sds = np.zeros(len(genes), dtype=float)
    means[measured] = reference[measured].mean(axis=1)
    sds[measured] = reference[measured].std(axis=1, ddof=1)
    valid = measured & (sds > 0)
    z = np.zeros((len(genes), len(endpoint_idx)), dtype=float)
    z[valid] = (expr[valid][:, endpoint_idx] - means[valid, None]) / sds[valid, None]
    module_scores = np.column_stack([signs @ z[ix] / denominator for ix, signs, denominator in groups])
    scaled = (module_scores - np.array(parameters['scaler_mean'])) / np.array(parameters['scaler_scale'])
    p = expit(scaled @ np.array(parameters['coefficients'][0]) + parameters['intercept'][0])
    return roc_auc_score(y, p)

baseline = score(np.arange(expr.shape[1]))
rng = np.random.default_rng(20261006)
rows = []
for replicate in range(200):
    reference_idx = rng.choice(expr.shape[1], size=42, replace=False)
    rows.append({'replicate': replicate + 1, 'reference_n': 42,
                 'endpoint_auc': score(reference_idx)})
result = pd.DataFrame(rows)
result.to_csv(OUT / 'GSE249477_reference_composition_sensitivity.csv', index=False)
summary = {
    'seed': 20261006, 'replicates': 200, 'reference_n': 42,
    'endpoint_AD_n': 21, 'endpoint_CTL_n': 21,
    'full_62_reference_auc': baseline,
    'subset_median_auc': float(result.endpoint_auc.median()),
    'subset_5th_95th': result.endpoint_auc.quantile([.05, .95]).tolist(),
    'subset_min_max': [float(result.endpoint_auc.min()), float(result.endpoint_auc.max())],
    'proportion_auc_gt_0_6': float((result.endpoint_auc > .6).mean()),
    'interpretation': 'Random reference selection without diagnoses; descriptive assay-mapping sensitivity, not independent predictive validation.'
}
(OUT / 'GSE249477_reference_composition_sensitivity.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
