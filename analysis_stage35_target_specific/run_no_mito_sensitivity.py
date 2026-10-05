from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_absolute_error
from scipy.stats import spearmanr
from run_target_specific import prepare,fit_predict,TARGETS,SRC
ROOT=Path(__file__).resolve().parent
keys=['Donor ID','brain region']; p=pd.read_csv(SRC/'Astrocyte_paired_measured_pathology.csv'); o=pd.read_csv(SRC/'Immune_paired_measured_pathology.csv'); p=p[p.qc_pass].merge(o.loc[o.qc_pass,keys+['row_index']],on=keys,suffixes=('_astro','_micro'),validate='one_to_one').sort_values(keys).reset_index(drop=True)
donors=p['Donor ID'].to_numpy(); regions=p['brain region'].to_numpy(); cov=pd.read_csv(SRC/'SEA_AD_MTG_baseline_covariates.csv',index_col=0).loc[donors].to_numpy(); ys=np.log1p(p[TARGETS].to_numpy(float)); genes=[]; ann=[]
for cell,col in [('Astrocyte','row_index_astro'),('Immune','row_index_micro')]:
    annotation=pd.read_csv(SRC/(cell+'_official_genes.csv')); valid=~annotation['index'].str.startswith('MT-'); raw=np.load(SRC/(cell+'_donor_region_counts.npz'))['counts'][p[col].to_numpy()]; genes.append(np.log2(1+raw[:,valid.to_numpy()]/(raw.sum(axis=1,keepdims=True)/1e6)).astype(np.float32)); ann.append(annotation[valid].reset_index(drop=True))
ids=np.array(sorted(set(donors))); outputs=[]; logs=[]
for held in sorted(set(regions)):
    for fold,(_,tp) in enumerate(KFold(5,shuffle=True,random_state=2026).split(ids)):
        td=ids[tp]; tr=np.flatnonzero((regions!=held)&~np.isin(donors,td)); te=np.flatnonzero((regions==held)&np.isin(donors,td)); trainids=np.array(sorted(set(donors[tr]))); groups=np.array_split(sorted(set(regions[tr])),3); inner=[]
        for j,(_,vp) in enumerate(KFold(3,shuffle=True,random_state=2026+fold).split(trainids)):
            vd=trainids[vp]; vr=groups[j]; it=tr[~np.isin(donors[tr],vd)&~np.isin(regions[tr],vr)]; iv=tr[np.isin(donors[tr],vd)&np.isin(regions[tr],vr)]; inner.append((it,iv))
        for k,t in enumerate(TARGETS):
            y=ys[:,k]; cached=[(it,iv,*prepare(cov,genes,y,donors,regions,it,iv,'pooled')[:2]) for it,iv in inner]; scores=[]
            for alpha in [.01,.1,1.]:
                values=[]
                for it,iv,a,b in cached:
                    pred,_=fit_predict(a,b,y[it],donors[it],alpha,'pooled'); counts=pd.Series(donors[iv]).value_counts(); w=np.array([1/counts[d] for d in donors[iv]]); values.append(np.average(abs(pred-y[iv]),weights=w))
                scores.append(float(np.mean(values)))
            alpha=[.01,.1,1.][int(np.argmin(scores))]; a,b,selected=prepare(cov,genes,y,donors,regions,tr,te,'pooled'); pred,_=fit_predict(a,b,y[tr],donors[tr],alpha,'pooled'); syms=[v['index'].iloc[ii].tolist() for v,ii in zip(ann,selected)]; assert not any(symbol.startswith('MT-') for block in syms for symbol in block)
            logs.append(dict(region=held,fold=fold,target=t,alpha=alpha,selected_symbols=syms,train_donors=sorted(set(donors[tr])),test_donors=sorted(set(donors[te])),train_regions=sorted(set(regions[tr]))))
            for j,row in enumerate(te): outputs.append(dict(donor_id=donors[row],region=held,fold=fold,model='pooled_no_mito',target=t,observed=y[row],predicted=pred[j]))
        print('Sensitivity '+held+' fold '+str(fold),flush=True)
df=pd.DataFrame(outputs); df.to_csv(ROOT/'no_mito_predictions.csv',index=False); (ROOT/'no_mito_training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8'); result=[]
for (r,t),sub in df.groupby(['region','target']): result.append(dict(region=r,target=t,n=len(sub),mae=mean_absolute_error(sub.observed,sub.predicted),r2=r2_score(sub.observed,sub.predicted),spearman=float(spearmanr(sub.observed,sub.predicted).statistic)))
m=pd.DataFrame(result); m.to_csv(ROOT/'no_mito_metrics.csv',index=False); summary=m.groupby('target').agg(macro_mae=('mae','mean'),macro_r2=('r2','mean')).reset_index(); summary.to_csv(ROOT/'no_mito_summary.csv',index=False); print(summary.to_string(index=False))
