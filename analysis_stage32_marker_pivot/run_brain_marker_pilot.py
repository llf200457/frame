from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error
ROOT=Path(__file__).resolve().parent

class Features(BaseEstimator,TransformerMixin):
    def __init__(self,kind='expression'): self.kind=kind
    def fit(self,X,y=None):
        if self.kind!='metadata':
            g=X[:,6:]; self.indices_=np.argsort(np.var(g,axis=0),kind='stable')[-300:]
            self.scale_=StandardScaler().fit(g[:,self.indices_])
            self.pca_=PCA(n_components=3,svd_solver='full').fit(self.scale_.transform(g[:,self.indices_]))
        return self
    def transform(self,X):
        if self.kind=='metadata': return X[:,:6]
        pcs=self.pca_.transform(self.scale_.transform(X[:,6:][:,self.indices_]))
        return np.column_stack([X[:,:6],pcs]) if self.kind=='combined' else pcs

def main():
    meta=pd.read_csv(ROOT/'GSE106241_marker_metadata.csv',index_col=0)
    gene=pd.read_pickle(ROOT/'GSE106241_gene_logsignal.pkl').T.loc[meta.index]
    cov=np.column_stack([meta['age'],meta['sex (1= female, 2= male)'].eq(2).astype(float),meta['rna quality'].eq(2).astype(float),meta['rna quality'].eq(3).astype(float),meta['apoe (additive) (0=0 copies of apoee4, 1=1 copy of apoee4, 2=two copies of apoee4)'],meta['rna quality'].isna().astype(float)])
    ranks=gene.rank(axis=1,pct=True).to_numpy()
    X=np.column_stack([cov,ranks]); outputs=[]; metrics=[]; logs=[]
    for target in ['amyloid-beta 42 levels','braak stage','beta secretase activity']:
        good=meta[target].notna().to_numpy(); ids=meta.index[good]; Xt=X[good]; y=meta.loc[good,target].to_numpy(float)
        assert np.isfinite(y).all() and (y>=0).all()
        if target!='braak stage': y=np.log1p(y)
        for seed in [2026,2027,2028]:
            splits=list(KFold(5,shuffle=True,random_state=seed).split(Xt))
            for kind in ['mean','metadata','expression','combined']:
                pred=np.full(len(y),np.nan)
                for fold,(tr,te) in enumerate(splits):
                    if kind=='mean': fit=DummyRegressor().fit(Xt[tr],y[tr]); best={}
                    else:
                        pipe=Pipeline([('impute',SimpleImputer()),('features',Features(kind)),('scale',StandardScaler()),('ridge',Ridge())])
                        fit=GridSearchCV(pipe,{'ridge__alpha':[10.,100.,1000.]},cv=KFold(3,shuffle=True,random_state=seed+fold),scoring='neg_mean_absolute_error',n_jobs=1).fit(Xt[tr],y[tr]); best=fit.best_params_
                    pred[te]=fit.predict(Xt[te]); logs.append(dict(target=target,seed=seed,model=kind,fold=fold,train=ids[tr].tolist(),test=ids[te].tolist(),parameters=best))
                row=dict(target=target,seed=seed,model=kind,n=len(y),r2=r2_score(y,pred),mae=mean_absolute_error(y,pred),spearman=float(spearmanr(y,pred).statistic))
                if target=='braak stage': row['rounded_stage_mae']=mean_absolute_error(y,np.clip(np.rint(pred),0,6))
                metrics.append(row)
                for i in range(len(y)): outputs.append(dict(target=target,seed=seed,model=kind,gsm=ids[i],observed=y[i],predicted=pred[i]))
                print(json.dumps(row),flush=True)
    pd.DataFrame(metrics).to_csv(ROOT/'brain_marker_metrics.csv',index=False)
    pd.DataFrame(outputs).to_csv(ROOT/'brain_marker_predictions.csv',index=False)
    (ROOT/'brain_marker_training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['BRAIN_MARKER_PROTOCOL.md','GSE106241_family.soft.gz','GSE106241_gene_logsignal.pkl','GSE106241_marker_metadata.csv','run_brain_marker_pilot.py']}
    (ROOT/'brain_marker_run_manifest.json').write_text(json.dumps({'hashes':hashes,'unique_people':60,'repeat_seeds':[2026,2027,2028],'normalization':'depositor cohort-wide, followed by per-sample ranks and fold-local modeling'},indent=2),encoding='utf-8')
if __name__=='__main__': main()
