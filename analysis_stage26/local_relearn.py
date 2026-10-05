"""Exploratory target-only refit of eight frozen modules, assessed out of fold.

This is a diagnostic localization test, not an independent validation or a
replacement for the unchanged discovery model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE if (BASE / 'modules').is_dir() else BASE.parents[1]
OUT = ROOT / 'analysis_stage26'
frame = pd.read_csv(OUT / 'external_transport_predictions.csv')
frame = frame[(frame.cohort == 'GSE140829') & frame.status.isin(['AD', 'CTL'])].copy()
cols = [f'M{k:02d}' for k in range(1, 9)]
assert len(frame) == 427 and frame[cols].notna().all().all()
x = frame[cols].to_numpy()
y = frame.status.eq('AD').to_numpy().astype(int)
frozen = frame.prob_AD.to_numpy()
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260929)
oof = np.full(len(frame), np.nan)
foldid = np.full(len(frame), -1)
coefficients = []
for k, (fit, test) in enumerate(folds.split(x, y)):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=20260929))
    model.fit(x[fit], y[fit])
    oof[test] = model.predict_proba(x[test])[:, 1]
    foldid[test] = k + 1
    coefficients.append(model[-1].coef_[0])
assert np.isfinite(oof).all() and (foldid > 0).all()
result = frame[['sample', 'status', 'prob_AD']].rename(columns={'prob_AD': 'frozen_prob_AD'})
result['local_module_oof_prob_AD'] = oof
result['fold'] = foldid
result.to_csv(OUT / 'GSE140829_module_refit_oof.csv', index=False)
coefs = pd.DataFrame(coefficients, columns=cols, index=np.arange(1, 6))
coefs.index.name = 'fold'
coefs.to_csv(OUT / 'GSE140829_module_refit_coefficients.csv')
rng = np.random.default_rng(20261002)
case = np.flatnonzero(y == 1)
control = np.flatnonzero(y == 0)
deltas = []
for _ in range(2000):
    ix = np.r_[rng.choice(case, len(case), replace=True),
               rng.choice(control, len(control), replace=True)]
    deltas.append([roc_auc_score(y[ix], oof[ix]) - roc_auc_score(y[ix], frozen[ix]),
                   brier_score_loss(y[ix], oof[ix]) - brier_score_loss(y[ix], frozen[ix])])
deltas = np.asarray(deltas)
summary = {
    'analysis': 'target-cohort five-fold OOF refit on the same eight frozen modules',
    'n': len(y), 'AD_n': int(y.sum()), 'CTL_n': int((1-y).sum()),
    'frozen_external_auc': roc_auc_score(y, frozen),
    'frozen_external_ap': average_precision_score(y, frozen),
    'frozen_external_brier': brier_score_loss(y, frozen),
    'target_only_oof_auc': roc_auc_score(y, oof),
    'target_only_oof_ap': average_precision_score(y, oof),
    'target_only_oof_brier': brier_score_loss(y, oof),
    'delta_oof_minus_frozen_auc': roc_auc_score(y, oof)-roc_auc_score(y, frozen),
    'delta_auc_bootstrap_ci': np.quantile(deltas[:, 0], [.025, .975]).tolist(),
    'delta_oof_minus_frozen_brier': brier_score_loss(y, oof)-brier_score_loss(y, frozen),
    'delta_brier_bootstrap_ci': np.quantile(deltas[:, 1], [.025, .975]).tolist(),
    'fold_coefficient_sign_consistency': {name: int((np.sign(coefs[name]) == np.sign(coefs[name].median())).sum())
                                          for name in cols},
    'design_note': 'GSE140829 labels train each fold-specific model. OOF estimates local learnability only; they are not an independent validation of the refit model.'
}
(OUT / 'module_refit_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
print(coefs.to_string(float_format=lambda x: f'{x:.3f}'))
