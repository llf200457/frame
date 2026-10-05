"""Suggestion 6: post-review sensitivity analyses, never primary-model selection.
Run from project Python. Outputs preserve individual predictions and all settings.
"""
from pathlib import Path
import hashlib, json, re, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from neuroCombat import neuroCombat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'metadata/stage22'; OUT.mkdir(parents=True,exist_ok=True)
RES=Path(__file__).parent/'resources'
SEED=20260928
COLS=[f'M{i:02d}' for i in range(1,9)]
threadpool_limits(1)

def model(seed=2025,C=30):
    return make_pipeline(StandardScaler(),LogisticRegression(penalty='elasticnet',solver='saga',C=C,l1_ratio=.5,max_iter=20000,random_state=seed))

def metric(y,p):
    return {'auc':roc_auc_score(y,p),'brier':brier_score_loss(y,p),'ap':average_precision_score(y,p)}

def ci(y,p,n=2000):
    rng=np.random.default_rng(SEED); vals=[]
    a=np.flatnonzero(y==1); b=np.flatnonzero(y==0)
    for _ in range(n):
        ix=np.r_[rng.choice(a,len(a)),rng.choice(b,len(b))]; vals.append(roc_auc_score(y[ix],p[ix]))
    lo,hi=np.quantile(vals,[.025,.975]); return {'auc_low':lo,'auc_high':hi}

def delong(y,p0,p1):
    # Paired nonparametric covariance from case/control placement values; ties count 1/2.
    arr=[]; brr=[]; auc=[]
    for p in [p0,p1]:
        a=p[y==1]; b=p[y==0]; h=(a[:,None]>b).astype(float)+.5*(a[:,None]==b)
        arr.append(h.mean(axis=1)); brr.append(h.mean(axis=0)); auc.append(h.mean())
    cov=np.cov(arr)/sum(y==1)+np.cov(brr)/sum(y==0)
    var=float(np.array([-1,1])@cov@np.array([-1,1])); diff=auc[1]-auc[0]
    pval=2*norm.sf(abs(diff)/np.sqrt(var)) if var>0 else (1. if diff==0 else 0.)
    return {'delta_auc':diff,'delta_low':diff-1.96*np.sqrt(max(0,var)),'delta_high':diff+1.96*np.sqrt(max(0,var)),'delong_p':pval}

def load():
    t=pd.read_csv(ROOT/'modules/GSE63060_frozen_module_scores.csv',index_col=0)
    lab=pd.read_csv(ROOT/'processed/GSE63060_matrix_formal_labels.csv').set_index('matrix_sample')
    t=t.join(lab[['label_formal','gender:ch1','age:ch1']])
    e=pd.read_csv(ROOT/'metadata/stage20/GSE63061_relabelled_predictions_all_samples.csv').set_index('matrix_sample')
    lines=(ROOT/'metadata/GSE63061_series_metadata.txt').read_text(encoding='utf-8').splitlines()
    ids=re.findall(r'"([^"]*)"',next(l for l in lines if l.startswith('!Series_sample_id\t')))[0].split()
    gender=re.findall(r'"([^"]*)"',next(l for l in lines if l.startswith('!Sample_characteristics_ch1\t') and 'gender:' in l))
    gm=dict(zip(ids,[x.split(':',1)[1].strip() for x in gender])); e['sex']=e.geo_accession.map(gm)
    assert len(gm)==388 and e.sex.notna().all()
    return t,e

def run():
    t,e=load(); tl=t[t.label_formal.isin(['AD','Control'])]; ev=e[e.status.isin(['AD','CTL'])]
    x=tl[COLS].to_numpy(); y=tl.label_formal.eq('AD').astype(int).to_numpy()
    xe=ev[COLS].to_numpy(); ye=ev.status.eq('AD').astype(int).to_numpy()
    base=model().fit(x,y); p=base.predict_proba(xe)[:,1]
    assert np.max(abs(p-ev.prob_AD))<1e-10
    selftest=delong(ye,p,p); assert selftest['delong_p']==1 and abs(selftest['delta_auc'])<1e-12
    g1=pd.read_pickle(ROOT/'processed/GSE63060_gene_expression.pkl',compression='gzip')
    g2=pd.read_pickle(ROOT/'processed/GSE63061_gene_expression.pkl',compression='gzip')
    markers=pd.read_csv(RES/'genes.txt',sep='\t'); cells=[]; coverage=[]; scores={}
    shared=set(g1.index)&set(g2.index)
    for cell,part in markers.groupby('Cell population',sort=False):
        genes=sorted(set(part['HUGO symbols'])&shared)
        coverage.append({'cell':cell,'signature_n':part['HUGO symbols'].nunique(),'common_mapped_n':len(genes),'genes':';'.join(genes)})
        if genes: cells.append(cell)
    pd.DataFrame(coverage).to_csv(OUT/'cell_marker_coverage.csv',index=False)
    for ds,g in [('GSE63060',g1),('GSE63061',g2)]:
        s=pd.DataFrame({r['cell']:g.loc[r['genes'].split(';')].mean(axis=0) for r in coverage if r['common_mapped_n']})
        s.to_csv(OUT/f'{ds}_MCPcounter_scores.csv'); scores[ds]=s
    immune=[c for c in cells if c not in ['Endothelial cells','Fibroblasts']]
    corr=[]
    for ds,tab in [('GSE63060',tl),('GSE63061',ev)]:
        for mod in COLS:
            for cell in immune:
                rho,pv=spearmanr(tab[mod],scores[ds].loc[tab.index,cell]); corr.append({'dataset':ds,'module':mod,'cell':cell,'rho':rho,'p':pv})
    corr=pd.DataFrame(corr); corr['fdr']=multipletests(corr.p,method='fdr_bh')[1]; corr.to_csv(OUT/'module_cell_correlations.csv',index=False)
    preds={'Frozen':p}; rows=[]
    # Label-free cell-score nuisance regression fitted exclusively in discovery.
    for name,nuis in [('Neutrophil residual',['Neutrophils']),('Immune residual',immune)]:
        ct=scores['GSE63060'].loc[tl.index,nuis].to_numpy(); ce=scores['GSE63061'].loc[ev.index,nuis].to_numpy()
        reg=LinearRegression().fit(ct,x); xt=x-reg.predict(ct); xtest=xe-reg.predict(ce)
        preds[name]=model().fit(xt,y).predict_proba(xtest)[:,1]
    # Transductive comparators: all 388 target profiles available, no target diagnoses.
    extall=e[COLS].to_numpy(); discall=t[COLS].to_numpy()
    mean=discall.mean(axis=0); sd=discall.std(axis=0,ddof=1)
    aligned=(extall-extall.mean(axis=0))/extall.std(axis=0,ddof=1)*sd+mean
    index=e.index.get_indexer(ev.index)
    preds['Moment aligned']=base.predict_proba(aligned[index])[:,1]
    mem=pd.read_csv(ROOT/'modules/frozen_pca_modules.csv'); genes=mem.gene.tolist()
    comb=np.c_[g1.loc[genes].to_numpy(),g2.loc[genes].to_numpy()]
    cov=pd.DataFrame({'batch':[0]*g1.shape[1]+[1]*g2.shape[1]})
    adj=neuroCombat(dat=comb,covars=cov,batch_col='batch',eb=True,parametric=True)['data']
    adj1=pd.DataFrame(adj[:,:g1.shape[1]],index=genes,columns=g1.columns)
    adj2=pd.DataFrame(adj[:,g1.shape[1]:],index=genes,columns=g2.columns)
    mu=adj1.mean(axis=1); sig=adj1.std(axis=1,ddof=1)
    z1=adj1.sub(mu,axis=0).div(sig,axis=0); z2=adj2.sub(mu,axis=0).div(sig,axis=0)
    def score(z):
        return pd.DataFrame({m:(z.loc[a.gene].to_numpy()*np.sign(a.loading_signed.to_numpy())[:,None]).mean(axis=0) for m,a in mem.groupby('module')},index=z.columns)
    c1,c2=score(z1),score(z2)
    preds['ComBat']=model().fit(c1.loc[tl.index,COLS],y).predict_proba(c2.loc[ev.index,COLS])[:,1]
    for name,pr in preds.items():
        row={'method':name,**metric(ye,pr),**ci(ye,pr),**delong(ye,p,pr)}; rows.append(row)
    met=pd.DataFrame(rows); met['delong_p_holm']=np.r_[np.nan,multipletests(met.delong_p.iloc[1:],method='holm')[1]]; met.to_csv(OUT/'sensitivity_model_metrics.csv',index=False)
    pp=ev[['geo_accession','status','sex']].copy()
    for name,pr in preds.items():pp[name]=pr
    pp.to_csv(OUT/'sensitivity_model_predictions.csv')
    # Sex stratification is descriptive, no threshold or model selected by subgroup.
    mu=t[COLS].mean().to_numpy(); inv=np.linalg.pinv(np.cov(t[COLS],rowvar=False))
    def drift(a):return np.einsum('ij,jk,ik->i',a-mu,inv,a-mu)
    threshold=np.quantile(drift(t[COLS].to_numpy()),.95); de=drift(xe)
    sex=[]
    for label in ['Female','Male']:
        ix=ev.sex.eq(label).to_numpy(); sex.append({'sex':label,'n':sum(ix),'AD':sum(ye[ix]),'CTL':sum(ye[ix]==0),**metric(ye[ix],p[ix]),**ci(ye[ix],p[ix]),'drift_rate':np.mean(de[ix]>threshold)})
    pd.DataFrame(sex).to_csv(OUT/'sex_stratified_metrics.csv',index=False)
    # Retention curve: deterministic target-relative ordering, evaluated only, not optimized.
    order=np.argsort(de); retain=[]
    for f in np.arange(.2,1.01,.05):
        ix=order[:round(len(order)*f)]; retain.append({'retained_fraction':len(ix)/len(order),'n':len(ix),'AD':sum(ye[ix]),'CTL':sum(ye[ix]==0),'threshold':de[ix].max(),**metric(ye[ix],p[ix])})
    pd.DataFrame(retain).to_csv(OUT/'relative_drift_retention.csv',index=False)
    (OUT/'drift_audit.json').write_text(json.dumps({'discovery_threshold':threshold,'all_external_n':len(e),'all_external_alert_fraction':float(np.mean(drift(extall)>threshold))},indent=2))
    print('Cell, adaptation, subgroup and drift analyses complete',flush=True)
    # Calibration-size stress test, repeated stratified splits; archived representation conditional.
    records=[]; weights=[]
    for frac in [.1,.2,.3,.4]:
        for seed in range(2025,2045):
            fi,ca=train_test_split(np.arange(len(y)),test_size=frac,stratify=y,random_state=seed)
            m=model(seed).fit(x[fi],y[fi]); pc=m.predict_proba(x[ca])[:,1]; pe=m.predict_proba(xe)[:,1]
            nc=np.where(y[ca],1-pc,pc); rank=int(np.ceil((len(ca)+1)*.9)); q=np.sort(nc)[rank-1] if rank<=len(ca) else np.inf
            domain=make_pipeline(StandardScaler(),LogisticRegression(max_iter=10000,random_state=seed)).fit(np.r_[x[fi],extall],np.r_[np.zeros(len(fi)),np.ones(len(extall))])
            dp=domain.predict_proba(x[ca])[:,1]; raw=dp/np.clip(1-dp,1e-6,None)*len(fi)/len(extall); w=np.clip(raw,.05,20)
            oo=np.argsort(nc); qw=nc[oo][np.searchsorted(np.cumsum(w[oo])/w.sum(),.9)]
            if frac==.2 and seed==2025:
                pd.DataFrame({'sample':tl.index[ca],'y':y[ca],'nonconformity':nc,'raw_weight':raw,'clipped_weight':w}).to_csv(OUT/'calibration_weights_audit.csv',index=False)
            for name,qq in [('Ordinary',q),('Reweighted',qw)]:
                c0=pe<=qq;c1=1-pe<=qq;size=c0.astype(int)+c1.astype(int)
                records.append({'calibration_fraction':frac,'calibration_n':len(ca),'seed':seed,'method':name,'q':qq,'coverage':np.where(ye,c1,c0).mean(),'AD_coverage':c1[ye==1].mean(),'CTL_coverage':c0[ye==0].mean(),'empty_rate':np.mean(size==0),'both_rate':np.mean(size==2),'mean_set_size':size.mean(),'effective_n':w.sum()**2/(w*w).sum(),'weight_clip_low_fraction':np.mean(raw<.05),'weight_clip_high_fraction':np.mean(raw>20)})
    pd.DataFrame(records).to_csv(OUT/'conformal_size_sensitivity.csv',index=False)
    print('80 conformal splits complete',flush=True)
    provenance={'seed':SEED,'bootstrap_n':2000,'conformal_seeds':list(range(2025,2045)),'cell_method':'MCP-counter HUGO marker mean; shared markers; relative scores, not fractions','cell_source':'https://github.com/ebecht/MCPcounter','combat_source':'https://github.com/Jfortin1/neuroCombat','target_adaptation':'all 388 target profiles; no target diagnosis used for fitting','primary_model':'unchanged; all additions post-review sensitivity analyses','resources_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RES.iterdir() if p.is_file()}}
    (OUT/'analysis_settings.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(met.to_string(index=False));print(pd.DataFrame(sex).to_string(index=False))

if __name__=='__main__':run()
