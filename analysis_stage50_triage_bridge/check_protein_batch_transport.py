"""Exploratory robustness of the prespecified two-module protein pilot.

Batch labels are reconstructed only for source sample IDs with a unique
nonexcluded batch record. Ambiguous repeated aliquots are retained in the
main donor pilot but not forced into a fabricated batch here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV,GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_protein_pilot import ProteinFeatures

ROOT=Path(__file__).resolve().parent
threadpool_limits(1)
bundle=np.load(ROOT/'paired_microglia_pseudobulk.npz')
gene_names=tuple(bundle['genes'].tolist())
donors=bundle['donors'].tolist()
clinical=pd.read_csv(ROOT/'paired_donor_protein_clinical_wb.csv',dtype={'donor':str}).set_index('donor').loc[donors]
quality=pd.read_csv(ROOT/'protein_pilot_rna_quality_covariates.csv',index_col=0,dtype={'donor':str})
quality.index=quality.index.astype(str);quality=quality.loc[donors]
valid=clinical.metadata_batch.notna().to_numpy()
groups=clinical.loc[valid,'metadata_batch'].to_numpy()
x=np.column_stack([bundle['log2cpm'],quality.to_numpy()])[valid]
y=clinical.loc[valid,'NFkB_log1p_ratio_mean'].to_numpy()
names=clinical.index[valid].tolist()
members=pd.read_csv(ROOT.parent/'modules/frozen_pca_modules.csv')
membership=tuple((r.gene,r.module,float(np.sign(r.loading_signed))) for r in members.itertuples())
outer=GroupKFold(min(5,len(set(groups))))
records,metrics=[],[]
for mode in ['mean','quality','module2']:
    pred=np.zeros(len(y))
    for fold,(train,test) in enumerate(outer.split(x,y,groups),1):
        assert not set(groups[train])&set(groups[test])
        if mode=='mean':
            model=DummyRegressor().fit(x[train,:1],y[train]);pred[test]=model.predict(x[test,:1])
        else:
            pipeline=Pipeline([('features',ProteinFeatures(mode,gene_names,membership)),('impute',SimpleImputer(strategy='median')),
                               ('scale',StandardScaler()),('ridge',Ridge())])
            inner=GroupKFold(min(3,len(set(groups[train]))))
            model=GridSearchCV(pipeline,{'ridge__alpha':[1.,10.,100.]},cv=list(inner.split(x[train],y[train],groups[train])),scoring='neg_mean_absolute_error',error_score='raise',n_jobs=1)
            model.fit(x[train],y[train]);pred[test]=model.predict(x[test])
        records.extend({'mode':mode,'fold':fold,'donor':names[i],'source_batch':groups[i],'true':y[i],'pred':pred[i]} for i in test)
    metrics.append({'mode':mode,'n':len(y),'source_batch_groups':len(set(groups)),'R2':r2_score(y,pred),'MAE':mean_absolute_error(y,pred)})
pd.DataFrame(records).to_csv(ROOT/'protein_batch_holdout_oof.csv',index=False)
pd.DataFrame(metrics).to_csv(ROOT/'protein_batch_holdout_metrics.csv',index=False)
(ROOT/'protein_batch_holdout_manifest.json').write_text(json.dumps({'date':'2026-10-02','exploratory_after_main_pilot':True,
 'batch_source':'unique source metadata date per donor; barcode-level batch labels absent from processed object',
 'n':len(y),'groups':len(set(groups)),'batch_and_donor_disjoint':True,'independent_paired_study_validation':False,
 'ambiguous_batch_donors_excluded_only_from_batch_audit':clinical.index[~valid].tolist()},ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(metrics).to_string(index=False))
