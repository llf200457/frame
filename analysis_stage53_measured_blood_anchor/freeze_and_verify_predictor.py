"""Freeze a factor predictor and independently verify saved OOF statistics."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding="utf-8")
threadpool_limits(limits=1)
OUT = Path(__file__).resolve().parent
DATA = OUT.parent
protocol = json.loads((OUT / 'lock_transport_predictor_protocol.json').read_text(encoding='utf-8'))
meta = pd.read_csv(OUT / 'paired_clinical_metadata.csv', index_col='sample', keep_default_na=False)
rna = pd.read_csv(OUT / 'raw/GSE157103_genes.tpm.tsv.gz', sep='\t', index_col=0)
members = pd.read_csv(DATA / 'modules/frozen_pca_modules.csv')
names = meta.index.tolist()
genes = [g for g in members.gene if g in rna.index]
logx = np.log2(rna.loc[genes, names].to_numpy(float).T + 1)
gene_mean, gene_sd = logx.mean(axis=0), logx.std(axis=0, ddof=1)
gene_sd[gene_sd == 0] = 1.0
lookup = members.set_index('gene')
mods = sorted(members.module.unique())
weight = np.zeros((len(genes), len(mods)))
for j, mod in enumerate(mods):
    mask = lookup.loc[genes, 'module'].eq(mod).to_numpy()
    weight[mask, j] = np.sign(lookup.loc[genes, 'loading_signed'].to_numpy()[mask]) / members.module.eq(mod).sum()
feature_names = ['age', 'male', 'sex_missing'] + mods
x = np.column_stack([meta[['age', 'male', 'sex_missing']].to_numpy(float), ((logx-gene_mean)/gene_sd) @ weight])
scaler = StandardScaler().fit(x)
y = meta.target_neutrophils_percent.to_numpy(float)
fit = Ridge(alpha=protocol['alpha']).fit(scaler.transform(x), y)
modelpath = OUT / 'locked_neutrophil_factor_predictor.npz'
np.savez_compressed(modelpath, genes=np.asarray(genes), gene_mean=gene_mean, gene_sd=gene_sd,
                    module_weights=weight, feature_names=np.asarray(feature_names),
                    feature_mean=scaler.mean_, feature_scale=scaler.scale_, coefficients=fit.coef_,
                    intercept=np.asarray(fit.intercept_), alpha=np.asarray(protocol['alpha']))
parameters = dict(training_n=len(y), target='measured neutrophil percent', feature_names=feature_names,
                  coefficients=fit.coef_.tolist(), intercept=float(fit.intercept_), alpha=protocol['alpha'],
                  model_file=modelpath.name, model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),
                  independent_validation=False, measured_protein_inputs=False,
                  original_AD_classifier_changed=False)
(OUT / 'locked_predictor_parameters.json').write_text(json.dumps(parameters,indent=2),encoding='utf-8')

# Independent replay from raw expression and fixed folds at alpha10, without
# calling the pilot's feature functions or model-scoring functions.
oof = pd.read_csv(OUT / 'out_of_fold_predictions.csv').set_index('sample').loc[names]
folds = oof.outer_fold.to_numpy(int)
fixed_oof = np.full(len(y), np.nan)
for fold in np.unique(folds):
    a, b = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    gm, gs = logx[a].mean(axis=0), logx[a].std(axis=0, ddof=1)
    gs[gs == 0] = 1.0
    xa = np.column_stack([meta[['age','male','sex_missing']].to_numpy(float)[a], ((logx[a]-gm)/gs) @ weight])
    xb = np.column_stack([meta[['age','male','sex_missing']].to_numpy(float)[b], ((logx[b]-gm)/gs) @ weight])
    scaling = StandardScaler().fit(xa)
    prediction = Ridge(alpha=10.0).fit(scaling.transform(xa),y[a]).predict(scaling.transform(xb))
    fixed_oof[b] = np.clip(prediction,0,100)
manual_fixed_r2 = 1.0 - np.sum((y-fixed_oof)**2)/np.sum((y-y.mean())**2)
pred = oof.age_sex_frozen8.to_numpy(float)
manual_r2 = 1.0 - np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
manual_mae = np.abs(y-pred).mean()
result = json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
assert abs(manual_r2-result['primary_result']['R2']) < 1e-12
assert abs(manual_mae-result['primary_result']['MAE']) < 1e-12
assert abs(manual_fixed_r2-result['fixed_alpha10_observed_R2']) < 1e-12
assert np.array_equal(y,oof.observed_neutrophils_percent.to_numpy(float))
# A reloaded model must reproduce arithmetic evaluation on training inputs;
# this checks serialization, not training-set performance.
saved = np.load(modelpath, allow_pickle=False)
manual_values = ((x-saved['feature_mean'])/saved['feature_scale']) @ saved['coefficients'] + saved['intercept']
replay_error = float(np.max(np.abs(manual_values-fit.predict(scaler.transform(x)))))
assert replay_error < 1e-10
verification = dict(sample_labels_match=True, manual_OOF_R2=manual_r2, manual_OOF_MAE=manual_mae,
                    manually_reconstructed_fixed_alpha10_R2=float(manual_fixed_r2),
                    serialization_max_error=replay_error,
                    preserved_files_unchanged=all(hashlib.sha256((DATA/p).read_bytes()).hexdigest()==h for p,h in result['preserved_before'].items()))
assert verification['preserved_files_unchanged']
(OUT/'independent_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2),flush=True)
