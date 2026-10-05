from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import sys,json,hashlib,warnings,argparse,time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import pairwise_distances,r2_score
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'analysis_stage39_tau_panel'))
from run_panel import dataset,design,predict
MODELS=['genes_ridge','random_forest','svr','joint_rbf','additive_kernel','interaction_kernel','astro_kernel','micro_kernel','late_fusion']
TARGETS=['Tau_panel','Tau_DFC','Tau_MEC']

def grid(kind):
    if kind=='genes_ridge':return [10.,100.,1000.]
    if kind=='random_forest':return [2,5,10]
    if kind=='svr':return [.1,1.,10.]
    return [.01,.1,1.]

def matrices(a,b,kind):
    ca,aa,ma=a[:,:30],a[:,30:40],a[:,40:50]
    cb,ab,mb=b[:,:30],b[:,30:40],b[:,40:50]
    def rb(x,z):return np.exp(-pairwise_distances(z,x,metric='sqeuclidean')/x.shape[1])
    if kind=='joint_rbf':return rb(a,a),rb(a,b)
    denom=max(float(np.mean(np.sum(ca*ca,axis=1)/30)),1e-8)
    kc=ca@ca.T/30/denom;kct=cb@ca.T/30/denom
    ka,kat=rb(aa,aa),rb(aa,ab);km,kmt=rb(ma,ma),rb(ma,mb)
    if kind=='astro_kernel':return kc+ka,kct+kat
    if kind=='micro_kernel':return kc+km,kct+kmt
    k=kc+.5*(ka+km);kt=kct+.5*(kat+kmt)
    if kind=='interaction_kernel':k=k+ka*km;kt=kt+kat*kmt
    return k,kt

def fit(pack,parameter,kind,seed):
    a,b,y,offset,_=pack; assert np.all(offset==0)
    mean=y.mean();sd=max(float(y.std()),1e-8);yy=(y-mean)/sd
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        if 'kernel' in kind or kind=='joint_rbf':
            k,kt=matrices(a,b,kind);v=kt@np.linalg.solve(k+parameter*np.eye(len(k)),yy)
        elif kind=='late_fusion':
            vv=[]
            for columns in [list(range(40)),list(range(30))+list(range(40,50))]:
                model=ElasticNet(alpha=parameter,l1_ratio=.5,max_iter=30000,tol=1e-7).fit(a[:,columns],yy);vv.append(model.predict(b[:,columns]))
            v=np.mean(vv,axis=0)
        else:
            if kind=='genes_ridge':model=Ridge(alpha=parameter)
            elif kind=='random_forest':model=RandomForestRegressor(n_estimators=128,min_samples_leaf=parameter,max_features=1.,random_state=seed,n_jobs=1)
            elif kind=='svr':model=SVR(C=parameter,epsilon=.1,gamma=1/a.shape[1])
            else:raise ValueError(kind)
            model.fit(a,yy);v=model.predict(b)
    if any(issubclass(z.category,ConvergenceWarning) for z in caught):raise RuntimeError('convergence failure')
    return v*sd+mean

def smoke(data):
    donors,cov,genes,ann,mi,ys,names=data;y=ys[:,names.index('Tau_panel')]
    tr,te=next(KFold(5,shuffle=True,random_state=2026).split(donors));pack=design(cov,genes,mi,y,tr,te,'genes')
    altered=y.copy();altered[te]=9999;other=design(cov,genes,mi,altered,tr,te,'genes')
    assert all(np.array_equal(x,z) for x,z in zip(pack[:4],other[:4]))
    eig=np.linalg.eigvalsh(matrices(pack[0],pack[0],'interaction_kernel')[0]);assert eig.min()>-1e-7
    oldlog=json.loads((ROOT.parent/'analysis_stage39_tau_panel/training_log.json').read_text());row=next(z for z in oldlog if z['target']=='Tau_panel' and z['model']=='genes' and z['seed']==2026 and z['fold']==0)
    pr,_=predict(pack,row['alpha'],'genes');old=pd.read_csv(ROOT.parent/'analysis_stage39_tau_panel/predictions.csv');old=old[(old.target=='Tau_panel')&(old.model=='genes')&(old.seed==2026)&(old.fold==0)].set_index('donor_id').loc[donors[te]].predicted.to_numpy()
    error=float(abs(pr-old).max());assert error<1e-10
    checks=[]
    for kind in MODELS:
        v=fit(pack,grid(kind)[1],kind,2026);assert np.isfinite(v).all();checks.append(dict(model=kind,n=len(v),finite=True))
    result=dict(reference_max_difference=error,test_outcome_perturbation_unchanged=True,interaction_min_eigenvalue=float(eig.min()),checks=checks)
    (ROOT/'smoke_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();data=dataset();smoke(data)
    if args.smoke:return
    donors,cov,genes,ann,mi,ys,names=data;rows=[];logs=[];start=time.time()
    for target in TARGETS:
      y=ys[:,names.index(target)]
      for seed in [2026,2027,2028]:
        for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(donors)):
          inner=[(tr[it],tr[iv]) for it,iv in KFold(3,shuffle=True,random_state=seed+fold).split(tr)]
          packs=[(iv,design(cov,genes,mi,y,it,iv,'genes')) for it,iv in inner];outer=design(cov,genes,mi,y,tr,te,'genes')
          for kind in MODELS:
            parameters=grid(kind);scores=[float(np.mean([abs(fit(p,val,kind,seed+fold)-y[iv]).mean() for iv,p in packs])) for val in parameters];parameter=parameters[int(np.argmin(scores))]
            v=fit(outer,parameter,kind,seed+fold)
            logs.append(dict(target=target,seed=seed,fold=fold,model=kind,parameter=parameter,inner_mae=scores,train_donors=donors[tr].tolist(),test_donors=donors[te].tolist(),inner_splits=[dict(train=donors[it].tolist(),validation=donors[iv].tolist()) for it,iv in inner],selected_genes=[dict(cell=c,ids=a.gene_ids.iloc[ii].tolist(),symbols=a['index'].iloc[ii].tolist()) for c,a,ii in zip(['Astrocyte','Micro-PVM'],ann,outer[-1])]))
            rows.extend(dict(donor_id=donors[j],target=target,seed=seed,fold=fold,model=kind,observed=float(y[j]),predicted=float(pr)) for j,pr in zip(te,v))
          pd.DataFrame(rows).to_csv(ROOT/'predictions.csv',index=False);(ROOT/'training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8');print(target,seed,fold,'done',round(time.time()-start,1),'s',flush=True)
    (ROOT/'run_manifest.json').write_text(json.dumps(dict(donors=len(donors),outer_fits=len(logs),predictors_including_inner=len(logs)*10+sum(z['model']=='late_fusion' for z in logs)*10,records=len(rows),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),elapsed_seconds=time.time()-start,python=sys.version),indent=2),encoding='utf-8')
if __name__=='__main__':main()
