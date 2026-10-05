from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import sys,json,hashlib,warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'analysis_stage37_signal_audit'))
from run_quality_audit import load
MODELS=['metadata','quality','genes','markers','orthogonal']
MARKERS=[['SERPINH1','CD44','WIF1','SLC1A2','GFAP'],['GPNMB','CD74','APOE','TREM2']]

def dataset():
    p,c,g,ann=load()
    t=p.pivot(index='Donor ID',columns='brain region',values='percent AT8 positive area')
    a=p.pivot(index='Donor ID',columns='brain region',values='percent 6e10 positive area')
    ids=t.index[t[['MTG','DFC','MEC']].notna().all(axis=1)&a[['MTG','DFC','MEC']].notna().all(axis=1)]
    ix=p.index[(p['brain region']=='MTG')&p['Donor ID'].isin(ids)].to_numpy()
    donors=p.loc[ix,'Donor ID'].to_numpy(); t=np.log1p(t.loc[donors]); a=np.log1p(a.loc[donors])
    ys=pd.DataFrame({'Tau_panel':t[['MTG','DFC','MEC']].mean(axis=1),'Tau_DFC':t.DFC,'Tau_MEC':t.MEC,'Amyloid_panel':a[['MTG','DFC','MEC']].mean(axis=1)})
    cov=c.iloc[ix].to_numpy(float); genes=[v[ix].astype(float) for v in g]
    mi=[]; coverage=[]
    for cell,z,aa,names in zip(['Astrocyte','Micro-PVM'],genes,ann,MARKERS):
      columns=[]
      for name in names:
        hits=np.flatnonzero(aa['index'].eq(name).to_numpy()); assert len(hits)==1
        j=int(hits[0]); columns.append(j);coverage.append(dict(cell=cell,symbol=name,gene_id=aa.gene_ids.iloc[j],nonzero_donors=int((z[:,j]>0).sum()),median_log2cpm=float(np.median(z[:,j]))))
      mi.append(columns)
    pd.DataFrame(coverage).to_csv(ROOT/'marker_coverage.csv',index=False)
    pd.concat([pd.Series(donors,name='donor_id'),pd.DataFrame(cov,columns=c.columns),ys.reset_index(drop=True)],axis=1).to_csv(ROOT/'analysis_cohort.csv',index=False)
    return donors,cov,genes,ann,mi,ys.to_numpy(),ys.columns.tolist()

def design(cov,genes,mi,y,tr,te,kind):
    cv=cov[:,:6] if kind=='metadata' else cov
    imp=SimpleImputer().fit(cv[tr]); sc=StandardScaler().fit(imp.transform(cv[tr])); ca=sc.transform(imp.transform(cv[tr])); cb=sc.transform(imp.transform(cv[te]))
    yt=y[tr]; offset=np.zeros(len(te)); selected=[]
    if kind=='orthogonal':
        baseline=Ridge(alpha=10.).fit(ca,yt);yt=yt-baseline.predict(ca);offset=baseline.predict(cb)
    aa=[];bb=[]
    if kind not in ['orthogonal']: aa.append(ca);bb.append(cb)
    if kind in ['genes','markers','orthogonal']:
      for z,marker in zip(genes,mi):
        if kind=='markers': ii=np.array(marker);ga=z[tr][:,ii];gb=z[te][:,ii]
        else:
          ii=np.argsort(np.var(z[tr],axis=0),kind='stable')[-1000:];ga=z[tr][:,ii];gb=z[te][:,ii]
          if kind=='orthogonal':
            nuisance=Ridge(alpha=10.).fit(ca,ga);ga=ga-nuisance.predict(ca);gb=gb-nuisance.predict(cb)
          gx=ga-ga.mean(axis=0);yy=yt-yt.mean();den=np.sqrt((gx*gx).sum(axis=0)*(yy*yy).sum());corr=np.divide(gx.T@yy,den,out=np.zeros(len(ii)),where=den>1e-12)
          take=np.argsort(abs(corr),kind='stable')[-10:];ii=ii[take];ga=ga[:,take];gb=gb[:,take]
        selected.append(ii.tolist());aa.append(ga);bb.append(gb)
    a=np.column_stack(aa);b=np.column_stack(bb);s=StandardScaler().fit(a)
    return s.transform(a),s.transform(b),yt,offset,selected

def predict(pack,alpha,kind):
    a,b,y,offset,selected=pack;mean=y.mean();sd=max(y.std(),1e-8)
    model=Ridge(alpha=alpha) if kind in ['metadata','quality'] else ElasticNet(alpha=alpha,l1_ratio=.5,max_iter=30000,tol=1e-7)
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always',ConvergenceWarning);model.fit(a,(y-mean)/sd)
    if any(issubclass(v.category,ConvergenceWarning) for v in caught):raise RuntimeError('Convergence failed')
    return model.predict(b)*sd+mean+offset,model

def nested(donors,cov,genes,ann,mi,y,target,seed,models=MODELS):
    output=[];logs=[]
    for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(donors)):
      assert not set(donors[tr])&set(donors[te]);inner=[(tr[it],tr[iv]) for it,iv in KFold(3,shuffle=True,random_state=seed+fold).split(tr)]
      for kind in models:
        grid=[10.,100.,1000.] if kind in ['metadata','quality'] else [.01,.1,1.]
        packs=[(iv,design(cov,genes,mi,y,it,iv,kind)) for it,iv in inner]
        score=[float(np.mean([np.mean(abs(predict(pack,alpha,kind)[0]-y[iv])) for iv,pack in packs])) for alpha in grid]
        alpha=grid[int(np.argmin(score))];pack=design(cov,genes,mi,y,tr,te,kind);pred,model=predict(pack,alpha,kind)
        logs.append(dict(seed=seed,fold=fold,target=target,model=kind,alpha=alpha,inner_mae=score,train_donors=donors[tr].tolist(),test_donors=donors[te].tolist(),inner_splits=[dict(train=donors[it].tolist(),validation=donors[iv].tolist()) for it,iv in inner],selected_genes=[dict(cell=c,symbols=a['index'].iloc[ii].tolist(),ids=a.gene_ids.iloc[ii].tolist()) for c,a,ii in zip(['Astrocyte','Micro-PVM'],ann,pack[-1])],coefficients=model.coef_.tolist(),intercept=float(model.intercept_)))
        output.extend(dict(donor_id=donors[j],target=target,model=kind,seed=seed,fold=fold,observed=float(y[j]),predicted=float(pr)) for j,pr in zip(te,pred))
      print(target,seed,fold,'complete',flush=True)
    return output,logs

def main():
    donors,cov,genes,ann,mi,ys,targets=dataset();output=[];logs=[]
    for k,target in enumerate(targets):
      for seed in [2026,2027,2028]:
        o,l=nested(donors,cov,genes,ann,mi,ys[:,k],target,seed);output+=o;logs+=l
        pd.DataFrame(output).to_csv(ROOT/'predictions.csv',index=False);(ROOT/'training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    (ROOT/'run_manifest.json').write_text(json.dumps(dict(n=len(donors),outer_fits=len(logs),seeds=[2026,2027,2028],targets=targets,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),python=sys.version),indent=2),encoding='utf-8')
if __name__=='__main__':main()
