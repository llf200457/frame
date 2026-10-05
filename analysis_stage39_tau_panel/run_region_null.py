from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'analysis_stage37_signal_audit'))
from run_quality_audit import load
from run_target_specific import prepare,fit_predict,weight
p,cov,genes,ann=load();donors=p['Donor ID'].to_numpy();regions=p['brain region'].to_numpy();y=np.log1p(p['percent AT8 positive area'].to_numpy(float));ids=np.array(sorted(set(donors)));outputs=[];logs=[]
def surrogate(tr):
 z=y.copy()
 for region in sorted(set(regions[tr])):
  ii=tr[regions[tr]==region];z[ii]=np.average(y[ii],weights=weight(donors[ii]))
 return z
for held in sorted(set(regions)):
 for fold,(_,tp) in enumerate(KFold(5,shuffle=True,random_state=2026).split(ids)):
  tr=np.flatnonzero((regions!=held)&~np.isin(donors,ids[tp]));te=np.flatnonzero((regions==held)&np.isin(donors,ids[tp]));assert not set(donors[tr])&set(donors[te]);trainids=np.array(sorted(set(donors[tr])));rgroups=np.array_split(sorted(set(regions[tr])),3);cache=[]
  for j,(_,vp) in enumerate(KFold(3,shuffle=True,random_state=2026+fold).split(trainids)):
   it=tr[~np.isin(donors[tr],trainids[vp])&~np.isin(regions[tr],rgroups[j])];iv=tr[np.isin(donors[tr],trainids[vp])&np.isin(regions[tr],rgroups[j])];sy=surrogate(it);a,b,_=prepare(cov.to_numpy(),genes,sy,donors,regions,it,iv,'pooled');cache.append((it,iv,a,b,sy))
  grid=[.01,.1,1.];score=[np.mean([np.average(abs(fit_predict(a,b,sy[it],donors[it],alpha,'pooled')[0]-y[iv]),weights=weight(donors[iv])) for it,iv,a,b,sy in cache]) for alpha in grid];alpha=grid[int(np.argmin(score))];sy=surrogate(tr);a,b,selected=prepare(cov.to_numpy(),genes,sy,donors,regions,tr,te,'pooled');pred,_=fit_predict(a,b,sy[tr],donors[tr],alpha,'pooled')
  logs.append(dict(held=held,fold=fold,alpha=alpha,inner_mae=score,train_donors=sorted(set(donors[tr])),test_donors=sorted(set(donors[te])),selected=selected,source_region_means={r:float(sy[tr[regions[tr]==r]][0]) for r in sorted(set(regions[tr]))}))
  outputs.extend(dict(donor_id=donors[j],region=held,fold=fold,seed=2026,observed=float(y[j]),predicted=float(v)) for j,v in zip(te,pred));print(held,fold,'null complete',flush=True)
 pd.DataFrame(outputs).to_csv(ROOT/'region_mean_null_predictions.csv',index=False)
(ROOT/'region_mean_null_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
