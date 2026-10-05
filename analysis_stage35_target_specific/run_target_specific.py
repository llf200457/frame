from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import json,hashlib,warnings,sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent; SRC=ROOT.parent/'analysis_stage33_seaad_pairing'
TARGETS=['percent 6e10 positive area','percent AT8 positive area']
def weight(d):
    counts=pd.Series(d).value_counts(); return np.array([1./counts[v] for v in d])
def weighted_var(x,w):
    mean=np.average(x,axis=0,weights=w); return np.average((x-mean)**2,axis=0,weights=w)
def demean(x,r,w):
    out=x.copy()
    for region in sorted(set(r)):
        mask=r==region; out[mask]-=np.average(x[mask],axis=0,weights=w[mask])
    return out
def prepare(cov,genes,y,donors,regions,tr,te,kind):
    w=weight(donors[tr]); imp=SimpleImputer().fit(cov[tr]); a=[imp.transform(cov[tr])]; b=[imp.transform(cov[te])]; selected=[]
    if kind!='metadata':
        for gene in genes:
            if kind=='pca':
                ids=np.argsort(np.var(gene[tr],axis=0),kind='stable')[-300:]; scaler=StandardScaler().fit(gene[tr][:,ids]); pc=PCA(3,svd_solver='full').fit(scaler.transform(gene[tr][:,ids])); a.append(pc.transform(scaler.transform(gene[tr][:,ids]))); b.append(pc.transform(scaler.transform(gene[te][:,ids])))
            else:
                candidate=np.argsort(weighted_var(gene[tr],w),kind='stable')[-1000:]; g=gene[tr][:,candidate].astype(float); t=y[tr].copy()
                if kind=='within': g=demean(g,regions[tr],w); t=demean(t[:,None],regions[tr],w)[:,0]
                g-=np.average(g,axis=0,weights=w); t-=np.average(t,weights=w)
                numerator=g.T@(w*t); denom=np.sqrt(np.sum(w[:,None]*g*g,axis=0)*np.sum(w*t*t)); corr=np.divide(numerator,denom,out=np.zeros_like(numerator),where=denom>1e-12)
                ids=candidate[np.argsort(abs(corr),kind='stable')[-10:]]; a.append(gene[tr][:,ids]); b.append(gene[te][:,ids]); selected.append(ids.tolist())
    a=np.column_stack(a); b=np.column_stack(b); scale=StandardScaler().fit(a,sample_weight=w); return scale.transform(a),scale.transform(b),selected
def fit_predict(a,b,y,donors,alpha,kind):
    w=weight(donors); mean=np.average(y,weights=w); sd=max(float(np.sqrt(np.average((y-mean)**2,weights=w))),1e-8)
    model=ElasticNet(alpha=alpha,l1_ratio=.5,max_iter=20000,tol=1e-7,selection='cyclic') if kind in ['pooled','within'] else Ridge(alpha=alpha)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning); model.fit(a,(y-mean)/sd,sample_weight=w)
    bad=[str(v.message) for v in caught if issubclass(v.category,ConvergenceWarning)]
    if bad: raise RuntimeError('Convergence failure: '+str(bad))
    return model.predict(b)*sd+mean,model
def main():
    keys=['Donor ID','brain region']; p=pd.read_csv(SRC/'Astrocyte_paired_measured_pathology.csv'); o=pd.read_csv(SRC/'Immune_paired_measured_pathology.csv'); p=p[p.qc_pass].merge(o.loc[o.qc_pass,keys+['row_index']],on=keys,suffixes=('_astro','_micro'),validate='one_to_one').sort_values(keys).reset_index(drop=True)
    assert len(p)==497
    meta=pd.read_csv(SRC/'SEA_AD_MTG_baseline_covariates.csv',index_col=0); donors=p['Donor ID'].to_numpy(); regions=p['brain region'].to_numpy(); cov=meta.loc[donors].to_numpy(); ys=np.log1p(p[TARGETS].to_numpy(float)); genes=[]; annotation=[]
    for cell,col in [('Astrocyte','row_index_astro'),('Immune','row_index_micro')]:
        raw=np.load(SRC/(cell+'_donor_region_counts.npz'))['counts'][p[col].to_numpy()]; genes.append(np.log2(1+raw/(raw.sum(axis=1,keepdims=True)/1e6)).astype(np.float32)); annotation.append(pd.read_csv(SRC/(cell+'_official_genes.csv')))
    ids=np.array(sorted(set(donors))); outputs=[]; logs=[]; innerlogs=[]
    for held in sorted(set(regions)):
        for fold,(_,testpos) in enumerate(KFold(5,shuffle=True,random_state=2026).split(ids)):
            td=ids[testpos]; tr=np.flatnonzero((regions!=held)&~np.isin(donors,td)); te=np.flatnonzero((regions==held)&np.isin(donors,td)); assert not set(donors[tr])&set(donors[te]) and held not in regions[tr]
            train_ids=np.array(sorted(set(donors[tr]))); rgroups=np.array_split(sorted(set(regions[tr])),3); inner=[]
            for j,(_,vpos) in enumerate(KFold(3,shuffle=True,random_state=2026+fold).split(train_ids)):
                vd=train_ids[vpos]; vr=rgroups[j]; it=tr[~np.isin(donors[tr],vd)&~np.isin(regions[tr],vr)]; iv=tr[np.isin(donors[tr],vd)&np.isin(regions[tr],vr)]; assert len(iv)>0 and not set(donors[it])&set(donors[iv]) and not set(regions[it])&set(regions[iv]); inner.append((it,iv)); innerlogs.append(dict(held_region=held,outer_fold=fold,inner_fold=j,train_donors=sorted(set(donors[it])),validation_donors=sorted(set(donors[iv])),train_regions=sorted(set(regions[it])),validation_regions=sorted(set(regions[iv]))))
            for k,target in enumerate(TARGETS):
                y=ys[:,k]
                for kind in ['metadata','pca','pooled','within']:
                    grid=[.01,.1,1.] if kind in ['pooled','within'] else [10.,100.,1000.]; cached=[(it,iv,*prepare(cov,genes,y,donors,regions,it,iv,kind)[:2]) for it,iv in inner]; scores=[]
                    for alpha in grid:
                        values=[]
                        for it,iv,a,b in cached:
                            prediction,_=fit_predict(a,b,y[it],donors[it],alpha,kind); values.append(float(np.average(abs(prediction-y[iv]),weights=weight(donors[iv]))))
                        scores.append(float(np.mean(values)))
                    alpha=grid[int(np.argmin(scores))]; a,b,selected=prepare(cov,genes,y,donors,regions,tr,te,kind); prediction,model=fit_predict(a,b,y[tr],donors[tr],alpha,kind)
                    selection=[dict(cell=cell,ids=anno.gene_ids.iloc[ii].tolist(),symbols=anno['index'].iloc[ii].tolist()) for cell,anno,ii in zip(['Astrocyte','Micro-PVM'],annotation,selected)]
                    logs.append(dict(held_region=held,fold=fold,target=target,model=kind,alpha=alpha,inner_mae=scores,selected_genes=selection,nonzero_coefficients=int(np.count_nonzero(model.coef_)),train_donors=sorted(set(donors[tr])),test_donors=sorted(set(donors[te])),train_regions=sorted(set(regions[tr])),test_records=len(te)))
                    for j,row in enumerate(te): outputs.append(dict(donor_id=donors[row],region=held,fold=fold,model=kind,target=target,observed=y[row],predicted=prediction[j]))
            pd.DataFrame(outputs).to_csv(ROOT/'predictions.csv',index=False); (ROOT/'training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8'); print(held+' fold '+str(fold)+' complete',flush=True)
    result=[]; df=pd.DataFrame(outputs)
    for (r,m,t),sub in df.groupby(['region','model','target']): result.append(dict(region=r,model=m,target=t,n=len(sub),mae=mean_absolute_error(sub.observed,sub.predicted),r2=r2_score(sub.observed,sub.predicted),spearman=float(spearmanr(sub.observed,sub.predicted).statistic)))
    metrics=pd.DataFrame(result); metrics.to_csv(ROOT/'metrics.csv',index=False); summary=metrics.groupby(['target','model']).agg(macro_mae=('mae','mean'),macro_r2=('r2','mean'),positive_r2_regions=('r2',lambda v:int((v>0).sum()))).reset_index(); summary.to_csv(ROOT/'macro_summary.csv',index=False)
    (ROOT/'inner_splits.json').write_text(json.dumps(innerlogs,indent=2),encoding='utf-8'); (ROOT/'run_manifest.json').write_text(json.dumps({'people':len(ids),'donor_region_records':len(p),'outer_model_fits':len(logs),'predictions':len(df),'outer_seed':2026,'convergence_warnings':0,'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version,'validation':'same-cohort exploratory; both inner and outer donor-and-region disjoint'},indent=2),encoding='utf-8'); print(summary.to_string(index=False),flush=True)
if __name__=='__main__': main()
