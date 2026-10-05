"""RNA-seq assay challenge using locked modules and label-free target scaling.

The clinical labels are read only after mapping and module scoring. The
discovery classifier is not trained or selected on this dataset.
"""
from pathlib import Path
import csv
import gzip
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE if (BASE / 'modules').is_dir() else BASE.parents[1]
OUT = ROOT / 'analysis_stage27'
eligibility = json.loads((OUT / 'eligibility_audit.json').read_text(encoding='utf-8'))
assert eligibility['eligible'] and eligibility['frozen_gene_coverage'] == 4686
meta = pd.read_csv(OUT / 'GSE249477_linked_metadata.csv').set_index('sample_id')
modules = pd.read_csv(ROOT / 'modules/frozen_pca_modules.csv')
param = json.loads((ROOT / 'overleaf_upload_route2_v7/source_data/stage22/primary_classifier_parameters.json').read_text())
path = ROOT / 'raw/GSE249477/GSE249477_raw_count_normalize_04-10-2025.csv.gz'
header = next(csv.reader(gzip.open(path, 'rt', encoding='utf-8', errors='replace')))
tpm_cols = [c for c in header if c.endswith(' - TPM')]
raw = pd.read_csv(path, usecols=['Name'] + tpm_cols, compression='gzip', low_memory=False).dropna(subset=['Name'])
valid_name = raw.Name.astype(str).str.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
complete = raw[tpm_cols].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
raw = raw.loc[valid_name & complete]
assert raw.Name.is_unique and len(tpm_cols) == 62
expr = np.log2(raw.set_index('Name')[tpm_cols].astype(float) + 1)
expr.columns = [c.split(' (GE)')[0] for c in tpm_cols]
assert set(expr.columns) == set(meta.index)
frozen = modules.gene.tolist()
shared = list(set(frozen) & set(expr.index))
assert len(shared) == 4686
gene_mean = expr.loc[shared].mean(axis=1)
gene_sd = expr.loc[shared].std(axis=1, ddof=1).replace(0, np.nan)
z = pd.DataFrame(0., index=frozen, columns=expr.columns)
z.loc[shared] = expr.loc[shared].sub(gene_mean, axis=0).div(gene_sd, axis=0).fillna(0)
scores = pd.DataFrame(index=expr.columns)
for module, group in modules.groupby('module'):
    scores[module] = np.sign(group.loading_signed.to_numpy()) @ z.loc[group.gene].to_numpy() / len(group)
scores = scores[param['features']]
x = (scores.to_numpy() - np.array(param['scaler_mean'])) / np.array(param['scaler_scale'])
prob = expit(x @ np.array(param['coefficients'][0]) + param['intercept'][0])
pred = scores.copy()
pred.insert(0, 'prob_AD', prob)
pred.insert(0, 'status', meta.loc[pred.index, 'status'].to_numpy())
pred.index.name = 'sample_id'
pred.to_csv(OUT / 'GSE249477_target_relative_predictions.csv')

def summarize(frame, endpoint, seed):
    y = frame.status.eq('AD').to_numpy().astype(int)
    p = frame.prob_AD.to_numpy()
    rng = np.random.default_rng(seed)
    case = np.flatnonzero(y == 1)
    other = np.flatnonzero(y == 0)
    aucs = []
    for _ in range(2000):
        ix = np.r_[rng.choice(case, len(case), replace=True),
                   rng.choice(other, len(other), replace=True)]
        aucs.append(roc_auc_score(y[ix], p[ix]))
    return {'endpoint': endpoint, 'n': len(frame), 'AD_n': int(y.sum()), 'non_AD_n': int((1-y).sum()),
            'auc': roc_auc_score(y,p), 'auc_bootstrap_ci': np.quantile(aucs,[.025,.975]).tolist(),
            'ap': average_precision_score(y,p), 'brier': brier_score_loss(y,p)}

binary = pred[pred.status.isin(['AD','CTL'])]
results = {'AD_vs_CTL': summarize(binary, 'AD versus cognitively normal control', 20261003),
           'AD_vs_CTL_plus_MCI': summarize(pred, 'AD versus control plus MCI', 20261004)}

def hedges(a,b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (1-3/(4*(na+nb)-9))*(np.mean(a)-np.mean(b))/pooled if pooled > 0 else np.nan

effects = {}
for name in ['M01','M07']:
    a = binary.loc[binary.status.eq('AD'),name].to_numpy()
    b = binary.loc[binary.status.eq('CTL'),name].to_numpy()
    rng = np.random.default_rng(20261005)
    bootstrap = [hedges(rng.choice(a,len(a),replace=True),rng.choice(b,len(b),replace=True)) for _ in range(2000)]
    effects[name] = {'g': hedges(a,b), 'bootstrap_ci': np.nanquantile(bootstrap,[.025,.975]).tolist()}

# Gene-effect concordance uses the same per-gene Hedges contrast, measured
# within each cohort, so RNA-seq and microarray units need not be pooled.
disc = pd.read_pickle(ROOT / 'processed/GSE63060_gene_expression.pkl', compression='gzip')
disc_meta = pd.read_csv(ROOT / 'processed/GSE63060_matrix_formal_labels.csv').set_index('matrix_sample')
ad_disc = disc_meta.index[disc_meta.label_formal.eq('AD')]
ctl_disc = disc_meta.index[disc_meta.label_formal.eq('Control')]
ad_new = meta.index[meta.status.eq('AD')]
ctl_new = meta.index[meta.status.eq('CTL')]
gene_rows = []
for name in shared:
    a = disc.loc[name, ad_disc].to_numpy(dtype=float)
    b = disc.loc[name, ctl_disc].to_numpy(dtype=float)
    c = expr.loc[name, ad_new].to_numpy(dtype=float)
    d = expr.loc[name, ctl_new].to_numpy(dtype=float)
    gene_rows.append((name, hedges(a,b), hedges(c,d)))
gene = pd.DataFrame(gene_rows, columns=['gene','discovery_g','GSE249477_g']).set_index('gene')
gene.to_csv(OUT / 'GSE249477_gene_effects.csv')
concordance = {}
for name, subset in [('all',gene),
                     ('M01',gene.loc[gene.index.intersection(modules.loc[modules.module.eq('M01'),'gene'])]),
                     ('M07',gene.loc[gene.index.intersection(modules.loc[modules.module.eq('M07'),'gene'])])]:
    pair = subset.dropna()
    concordance[name] = {'common_genes': len(pair),
                         'spearman_rho': float(spearmanr(pair.discovery_g,pair.GSE249477_g).statistic),
                         'same_direction_fraction': float((np.sign(pair.discovery_g)==np.sign(pair.GSE249477_g)).mean())}
results.update({'frozen_gene_coverage': len(shared), 'module_effects': effects,
                'gene_effect_concordance': concordance,
                'method': 'log2(TPM+1), gene-symbol mapping, label-free gene z-score across all 62 target profiles, frozen signed modules and classifier coefficients; no refitting',
                'interpretation': 'Cross-assay transductive challenge; probability values are not calibrated for direct clinical interpretation.'})
(OUT / 'GSE249477_results.json').write_text(json.dumps(results,indent=2), encoding='utf-8')
print(json.dumps(results,indent=2))
