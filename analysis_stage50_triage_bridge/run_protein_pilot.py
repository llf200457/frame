"""Finite donor-nested protein pilot linked to original blood module mapping.

All transforms, gene selection and ridge tuning are trained inside the folds.
This is a new-endpoint pilot, not a new algorithm or clinical blood validation.
"""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent
SEEDS=[20261002,20261003,20261004]
CONTEXT=['RELA','NFKB1','NFKB2','REL','RELB','NFKBIA','TNFAIP3','CHUK','IKBKB','IKBKG','TNFRSF1A','TRADD','RIPK1','TRAF2','CSF1R','CX3CR1','P2RY12','SPP1','TMEM163']
threadpool_limits(1)

class ProteinFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,mode='quality',gene_names=(),membership=()):
        self.mode,self.gene_names,self.membership=mode,gene_names,membership
    def fit(self,X,y=None):
        n=len(self.gene_names)
        self.mu_=X[:,:n].mean(axis=0)
        self.sd_=X[:,:n].std(axis=0,ddof=1)
        self.sd_[self.sd_<1e-12]=1
        lookup={g:i for i,g in enumerate(self.gene_names)}
        if self.mode in ['selected20','selected20_rna_only']:
            z=(X[:,:n]-self.mu_)/self.sd_
            centered=y-y.mean()
            score=np.abs(z.T@centered)/max(1e-12,np.linalg.norm(centered))
            self.selected_=np.argsort(score,kind='stable')[-20:]
        elif self.mode=='RELA': self.selected_=np.array([lookup['RELA']])
        elif self.mode=='context': self.selected_=np.array([lookup[g] for g in CONTEXT if g in lookup])
        elif self.mode.startswith('module'):
            self.weights_=np.zeros((n,2 if self.mode=='module2' else 8))
            chosen=['M01','M07'] if self.mode=='module2' else [f'M{i:02d}' for i in range(1,9)]
            for j,module in enumerate(chosen):
                rows=[r for r in self.membership if r[1]==module]
                for gene,_,sign in rows:
                    if gene in lookup: self.weights_[lookup[gene],j]=sign/len(rows)
        return self
    def transform(self,X):
        n=len(self.gene_names)
        q=X[:,n:]
        if self.mode=='quality': return q
        z=(X[:,:n]-self.mu_)/self.sd_
        a=z@self.weights_ if self.mode.startswith('module') else z[:,self.selected_]
        return a if self.mode=='selected20_rna_only' else np.column_stack([a,q])

def ccc(y,p):
    return float(2*np.mean((y-y.mean())*(p-p.mean()))/(np.var(y)+np.var(p)+(y.mean()-p.mean())**2))

def main():
    bundle=np.load(ROOT/'paired_microglia_pseudobulk.npz')
    donors,genes=bundle['donors'].tolist(),bundle['genes'].tolist()
    expr=bundle['log2cpm']
    clinical=pd.read_csv(ROOT/'paired_donor_protein_clinical_wb.csv',dtype={'donor':str}).set_index('donor').loc[donors]
    assert len(donors)==len(set(donors)) and np.isfinite(expr).all()
    y=clinical.NFkB_log1p_ratio_mean.to_numpy(float)
    q=pd.DataFrame({'age':pd.to_numeric(clinical.Age,errors='coerce'),
                    'male':clinical.Sex.astype(str).str.upper().eq('M').astype(float),
                    'PMI':pd.to_numeric(clinical.PostmortemInterval,errors='coerce'),
                    'RIN':pd.to_numeric(clinical.JoinedRIN,errors='coerce'),
                    'log_nuclei':np.log1p(clinical.microglia_n),
                    'log_UMI_per_nucleus':np.log1p(clinical.n_counts_mean),
                    'mitochondrial_fraction':clinical.frac_mito_mean})
    x=np.column_stack([expr,q.to_numpy(float)])
    members=pd.read_csv(DATA/'modules/frozen_pca_modules.csv')
    membership=tuple((r.gene,r.module,float(np.sign(r.loading_signed))) for r in members.itertuples())
    modes=['mean','quality','RELA','module2','module8','context','selected20','selected20_rna_only']
    predictions,metrics,fitlog=[],[],[]
    for seed in SEEDS:
        splits=list(KFold(5,shuffle=True,random_state=seed).split(x))
        for mode in modes:
            p=np.zeros(len(y))
            for fold,(train,test) in enumerate(splits,1):
                if mode=='mean':
                    fitted=DummyRegressor().fit(x[train,:1],y[train])
                    p[test]=fitted.predict(x[test,:1]);best=None
                else:
                    pipeline=Pipeline([('features',ProteinFeatures(mode,tuple(genes),membership)),('impute',SimpleImputer(strategy='median')),
                                       ('scale',StandardScaler()),('ridge',Ridge())])
                    fitted=GridSearchCV(pipeline,{'ridge__alpha':[1.,10.,100.]},cv=KFold(3,shuffle=True,random_state=seed+fold),scoring='neg_mean_absolute_error',n_jobs=1,error_score='raise')
                    fitted.fit(x[train],y[train]);p[test]=fitted.predict(x[test]);best=fitted.best_params_['ridge__alpha']
                fitlog.append({'seed':seed,'mode':mode,'fold':fold,'train_donors':len(train),'test_donors':len(test),'alpha':best})
            metrics.append({'seed':seed,'mode':mode,'n':len(y),'R2':r2_score(y,p),'MAE':mean_absolute_error(y,p),'CCC':ccc(y,p),'spearman_rho':spearmanr(y,p).statistic})
            predictions.extend({'seed':seed,'mode':mode,'donor':d,'true_NFkB':float(yy),'pred_NFkB':float(pp)} for d,yy,pp in zip(donors,y,p))
            print('Protein pilot',seed,mode,'R2',round(r2_score(y,p),3),flush=True)
    metrics=pd.DataFrame(metrics)
    pred=pd.DataFrame(predictions)
    metrics.to_csv(ROOT/'protein_pilot_metrics_per_seed.csv',index=False)
    pred.to_csv(ROOT/'protein_pilot_donor_oof.csv',index=False)
    pd.DataFrame(fitlog).to_csv(ROOT/'protein_pilot_folds.csv',index=False)
    summary=metrics.groupby('mode')[['R2','MAE','CCC']].mean().reset_index()
    quality=pred.loc[pred['mode']=='quality'].pivot(index='donor',columns='seed',values='pred_NFkB').loc[donors].to_numpy()
    draws=[];rng=np.random.default_rng(SEEDS[0])
    for mode in modes:
        pp=pred.loc[pred['mode']==mode].pivot(index='donor',columns='seed',values='pred_NFkB').loc[donors].to_numpy()
        delta=np.mean(np.abs(y[:,None]-quality)-np.abs(y[:,None]-pp),axis=1)
        boot=np.array([delta[rng.choice(len(y),len(y),replace=True)].mean() for _ in range(1000)])
        draws.append({'mode':mode,'MAE_gain_over_quality':float(delta.mean()),'gain_low':float(np.quantile(boot,.025)),
                      'gain_high':float(np.quantile(boot,.975)),'conditional_bootstrap_not_new_cohort':True})
    summary=summary.merge(pd.DataFrame(draws),on='mode')
    summary.to_csv(ROOT/'protein_pilot_summary.csv',index=False)
    # Association of the two existing module gene maps with measured protein.
    mapper=ProteinFeatures('module2',tuple(genes),membership).fit(x,y)
    module=mapper.transform(x)[:,:2]
    controls=np.column_stack([np.ones(len(y)),q[['age','male','RIN','log_UMI_per_nucleus']].fillna(q.median()).to_numpy(),
                              pd.get_dummies(clinical.DiseaseState,drop_first=True).to_numpy(float)])
    associations=[]
    for j,name in enumerate(['M01','M07']):
        a,b=pd.Series(module[:,j]).rank().to_numpy(),pd.Series(y).rank().to_numpy()
        ar=a-controls@np.linalg.lstsq(controls,a,rcond=None)[0]
        br=b-controls@np.linalg.lstsq(controls,b,rcond=None)[0]
        rho=float(np.corrcoef(ar,br)[0,1]);df=len(y)-np.linalg.matrix_rank(controls)-1
        pv=float(2*t.sf(abs(rho)*np.sqrt(df/max(1e-12,1-rho*rho)),df))
        associations.append({'module':name,'raw_spearman':spearmanr(module[:,j],y).statistic,'partial_rank_r':rho,'p':pv,'p_bonferroni_2':min(1,2*pv),'n':len(y),
                             'interpretation':'Frozen gene membership/signs with within-RNA-study normalization; not a blood clinical probability'})
    pd.DataFrame(associations).to_csv(ROOT/'blood_module_real_protein_bridge.csv',index=False)
    q.to_csv(ROOT/'protein_pilot_rna_quality_covariates.csv')
    manifest={'date':'2026-10-02','n_unique_donors':len(donors),'nested_outer_folds':5,'inner_folds':3,'seeds':SEEDS,
              'target':'real measured NFkB/H3 mean per-nucleus log1p ratio; not an RNA pathway score',
              'context_genes':CONTEXT,'model_modes':modes,'diagnosis_not_model_input':True,'H3_and_NFkB_ADT_not_model_input':True,
              'true_donor_disjoint_out_of_fold':True,'held_donor_excluded_from_gene_selection_scaling_and_tuning':True,
              'batch_and_age_dependence_not_yet_fully_resolved':True,'independent_paired_protein_study_validation':False,
              'original_blood_clinical_model_updated':False,'novel_algorithm_trained':False,'protein_pilot_trained':True,
              'whole_file_md5_verified':False,'verified_range_blocks_used':True,
              'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'protein_pilot_run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(pd.DataFrame(associations).to_string(index=False))

if __name__=='__main__': main()
