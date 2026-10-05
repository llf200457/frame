"""Diagnosis-blind normalization-reference sensitivity for GSE248417."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

root=Path(__file__).resolve().parents[1]
out=root/'analysis_stage28'
meta=pd.read_csv(out/'GSE248417_linked_metadata.csv')
modules=pd.read_csv(root/'modules/frozen_pca_modules.csv')
params=json.loads((root/'metadata/stage22/primary_classifier_parameters.json').read_text())
sample=meta['sample'].tolist(); y=meta.status.eq('AD').to_numpy(dtype=int)
raw=pd.read_csv(out/'GSE248417_ADvsControl.mRNA.gene_level.deg.txt.gz',
                sep='\t',usecols=['gene_name']+sample,compression='gzip')
expr=raw.groupby('gene_name',sort=True)[sample].mean().astype(float)
shared=modules.gene[modules.gene.isin(expr.index)].tolist()
assert len(shared)==4851 and len(sample)==98
x=np.log2(expr.loc[shared].to_numpy()+1)
gene_index={g:i for i,g in enumerate(shared)}
groups=[]
for name in params['features']:
    group=modules[modules.module.eq(name)]
    ix=np.array([gene_index[g] for g in group.gene if g in gene_index],int)
    signs=np.array([np.sign(v) for g,v in zip(group.gene,group.loading_signed) if g in gene_index])
    groups.append((ix,signs,len(group)))
rng=np.random.default_rng(20261011)
rows=[]
for rep in range(200):
    ref=rng.choice(x.shape[1],size=70,replace=False)
    mu=x[:,ref].mean(axis=1)
    sd=x[:,ref].std(axis=1,ddof=1)
    z=np.divide(x-mu[:,None],sd[:,None],out=np.zeros_like(x),where=sd[:,None]>0)
    feature=np.column_stack([(signs@z[ix])/denom for ix,signs,denom in groups])
    xx=(feature-np.array(params['scaler_mean']))/np.array(params['scaler_scale'])
    score=xx@np.array(params['coefficients'][0])+params['intercept'][0]
    rows.append({'replicate':rep+1,'auc':roc_auc_score(y,score),
                 'reference_n':len(ref),'evaluation_n':len(sample)})
frame=pd.DataFrame(rows)
frame.to_csv(out/'GSE248417_reference_composition_sensitivity.csv',index=False)
summary={'replicates':200,'reference_n':70,'evaluation_n':98,'seed':20261011,
         'auc_median':float(frame.auc.median()),
         'auc_q05':float(frame.auc.quantile(.05)),
         'auc_q95':float(frame.auc.quantile(.95)),
         'fraction_auc_at_least_0_65':float((frame.auc>=.65).mean()),
         'design_note':'Unlabelled random reference subsets estimate per-gene mean and SD; all 98 fixed cases and controls are scored in every replicate. These quantiles measure transductive reference sensitivity, not sampling confidence intervals.'}
(out/'GSE248417_reference_composition_sensitivity.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
