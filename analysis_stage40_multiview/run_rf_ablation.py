from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from run_benchmark import dataset,design,fit
ROOT=Path(__file__).resolve().parent
def prepare(data,y,tr,te,kind):
 donors,cov,genes,ann,mi,_,_=data
 if kind=='rf_no_mt':
  masked=[~a['index'].fillna('').str.startswith('MT-').to_numpy() for a in ann];g=[z[:,m] for z,m in zip(genes,masked)];a=[v.loc[m].reset_index(drop=True) for v,m in zip(ann,masked)];pack=design(cov,g,mi,y,tr,te,'genes');return pack,[dict(cell=c,ids=v.gene_ids.iloc[ii].tolist(),symbols=v['index'].iloc[ii].tolist()) for c,v,ii in zip(['Astrocyte','Micro-PVM'],a,pack[-1])]
 k=0 if kind=='rf_astro20' else 1;z=genes[k];ii=np.argsort(np.var(z[tr],axis=0),kind='stable')[-1000:];a=z[tr][:,ii];g=a-a.mean(axis=0);t=y[tr]-y[tr].mean();den=np.sqrt((g*g).sum(axis=0)*(t*t).sum());r=np.divide(g.T@t,den,out=np.zeros(len(ii)),where=den>1e-12);ii=ii[np.argsort(abs(r),kind='stable')[-20:]];covpack=design(cov,genes,mi,y,tr,te,'quality');sc=StandardScaler().fit(z[tr][:,ii]);aa=np.column_stack([covpack[0],sc.transform(z[tr][:,ii])]);bb=np.column_stack([covpack[1],sc.transform(z[te][:,ii])]);pack=(aa,bb,y[tr],np.zeros(len(te)),[ii.tolist()]);return pack,[dict(cell=['Astrocyte','Micro-PVM'][k],ids=ann[k].gene_ids.iloc[ii].tolist(),symbols=ann[k]['index'].iloc[ii].tolist())]
def main():
 data=dataset();donors,cov,genes,ann,mi,ys,names=data;rows=[];logs=[]
 for target in ['Tau_panel','Tau_DFC','Tau_MEC']:
  y=ys[:,names.index(target)]
  for seed in [2026,2027,2028]:
   for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(donors)):
    inner=[(tr[it],tr[iv]) for it,iv in KFold(3,shuffle=True,random_state=seed+fold).split(tr)]
    for kind in ['rf_astro20','rf_micro20','rf_no_mt']:
     packs=[(iv,prepare(data,y,it,iv,kind)[0]) for it,iv in inner];grid=[2,5,10];scores=[float(np.mean([abs(fit(pack,v,'random_forest',seed+fold)-y[iv]).mean() for iv,pack in packs])) for v in grid];v=grid[int(np.argmin(scores))];pack,selection=prepare(data,y,tr,te,kind);pr=fit(pack,v,'random_forest',seed+fold)
     rows.extend(dict(donor_id=donors[j],target=target,seed=seed,fold=fold,model=kind,observed=float(y[j]),predicted=float(p)) for j,p in zip(te,pr));logs.append(dict(target=target,seed=seed,fold=fold,model=kind,parameter=v,inner_mae=scores,train_donors=donors[tr].tolist(),test_donors=donors[te].tolist(),inner_splits=[dict(train=donors[it].tolist(),validation=donors[iv].tolist()) for it,iv in inner],selected_genes=selection))
  print(target,'ablation done',flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'rf_ablation_predictions.csv',index=False);(ROOT/'rf_ablation_training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
if __name__=='__main__':main()
