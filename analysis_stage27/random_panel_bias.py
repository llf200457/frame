"""Exploratory random-gene negative control for AddNeuroMed discovery AUC.

No random panel is optimized or selected for final reporting. All panels use
the same 5-fold split and ordinary L2 logistic regression with training-fold
feature scaling, so the distribution describes nonspecific internal signal.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE if (BASE / 'modules').is_dir() else BASE.parents[1]
OUT = ROOT / 'analysis_stage27'
expr = pd.read_pickle(ROOT / 'processed/GSE63060_gene_expression.pkl', compression='gzip')
meta = pd.read_csv(ROOT / 'processed/GSE63060_matrix_formal_labels.csv').set_index('matrix_sample')
sample_ids = meta.index[meta.label_formal.isin(['AD','Control'])]
assert len(sample_ids) == 249 and set(sample_ids) <= set(expr.columns)
y = meta.loc[sample_ids, 'label_formal'].eq('AD').to_numpy().astype(int)
usable = expr.index[expr[sample_ids].notna().all(axis=1) & (expr[sample_ids].std(axis=1,ddof=1)>0)]
matrix = expr.loc[usable, sample_ids].to_numpy(dtype=np.float32).T
assert np.isfinite(matrix).all()
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=20261006).split(matrix,y))
rng = np.random.default_rng(20261006)
rows = []
panels = []
for panel in range(200):
    indices = rng.choice(matrix.shape[1], size=75, replace=False)
    panels.append(indices)
    x = matrix[:,indices]
    oof = np.empty(len(y),dtype=float)
    for fit,test in folds:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1,max_iter=1000,random_state=20261006))
        model.fit(x[fit],y[fit])
        oof[test] = model.predict_proba(x[test])[:,1]
    rows.append({'panel':panel+1,'auc':roc_auc_score(y,oof),
                 'genes':'|'.join(usable[indices])})
frame = pd.DataFrame(rows)
frame.to_csv(OUT / 'random_panel_auc.csv',index=False)
permuted = []
for panel, indices in enumerate(panels[:100]):
    shuffled = rng.permutation(y)
    oof = np.empty(len(y),dtype=float)
    split = StratifiedKFold(n_splits=5,shuffle=True,random_state=20261006+panel)
    x = matrix[:,indices]
    for fit,test in split.split(x,shuffled):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1,max_iter=1000,random_state=20261006))
        model.fit(x[fit],shuffled[fit])
        oof[test] = model.predict_proba(x[test])[:,1]
    permuted.append({'panel':panel+1,'auc':roc_auc_score(shuffled,oof)})
null = pd.DataFrame(permuted)
null.to_csv(OUT / 'random_panel_label_permutation_auc.csv',index=False)
summary = {'n_panels':200,'genes_per_panel':75,'source_gene_pool':len(usable),
           'discovery_AD_n':int(y.sum()),'discovery_CTL_n':int((1-y).sum()),
           'five_fold_oof_auc_quantiles':{
               f'q{int(q*100):02d}':float(frame.auc.quantile(q)) for q in [0,.05,.25,.5,.75,.95,1]},
           'fraction_panels_auc_at_least_0_75':float((frame.auc>=.75).mean()),
           'n_label_permutations':len(null),
           'label_permutation_auc_quantiles':{
               f'q{int(q*100):02d}':float(null.auc.quantile(q)) for q in [0,.05,.5,.95,1]},
           'design_note':'Exploratory negative control. Random panels are not externally validated, not selected and not equivalent to the original full nested representation validation.'}
(OUT / 'random_panel_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
