from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from run_benchmark import dataset,design,fit,predict
ROOT=Path(__file__).resolve().parent
def ccc(y,p):return float(2*np.mean((y-y.mean())*(p-p.mean()))/(y.var()+p.var()+(y.mean()-p.mean())**2))
def main():
 donors,cov,genes,ann,mi,ys,names=dataset();meta=pd.read_csv(ROOT.parent/'analysis_stage32_marker_pivot/SEA_AD_donor_metadata.csv').set_index('Donor ID').loc[donors];source=meta['Primary Study Name'].to_numpy();pure=meta['Secondary Study Name'].isna().to_numpy();rows=[];logs=[];metrics=[]
 pd.DataFrame({'donor_id':donors,'primary_source':source,'excluded_secondary_source':~pure}).to_csv(ROOT/'recruitment_source_map.csv',index=False)
 for held in ['ACT','ADRC Clinical Core']:
  tr=np.flatnonzero(pure&(source!=held));te=np.flatnonzero(pure&(source==held));assert not set(donors[tr])&set(donors[te]);assert len(tr)>=9 and len(te)>=9
  inner=[(tr[it],tr[iv]) for it,iv in KFold(3,shuffle=True,random_state=40201).split(tr)]
  for target in ['Tau_panel','Tau_DFC','Tau_MEC']:
   y=ys[:,names.index(target)]
   for model,kind,learner,grid in [('metadata_ridge','metadata','ridge',[10.,100.,1000.]),('quality_rf','quality','random_forest',[2,5,10]),('genes_en','genes','en',[.01,.1,1.]),('genes_rf','genes','random_forest',[2,5,10])]:
    packs=[(iv,design(cov,genes,mi,y,it,iv,kind)) for it,iv in inner]
    def infer(pack,v):
     if learner=='random_forest':return fit(pack,v,learner,40201)
     return predict(pack,v,'metadata' if learner=='ridge' else 'genes')[0]
    scores=[float(np.mean([abs(infer(pack,v)-y[iv]).mean() for iv,pack in packs])) for v in grid];val=grid[int(np.argmin(scores))];pack=design(cov,genes,mi,y,tr,te,kind);pr=infer(pack,val)
    rows.extend(dict(donor_id=donors[j],held_source=held,target=target,model=model,observed=float(y[j]),predicted=float(v),training_mean=float(y[tr].mean())) for j,v in zip(te,pr));metrics.append(dict(held_source=held,target=target,model=model,train_n=len(tr),test_n=len(te),mae=float(abs(pr-y[te]).mean()),r2=float(r2_score(y[te],pr)),ccc=ccc(y[te],pr),training_mean_mae=float(abs(y[tr].mean()-y[te]).mean())))
    logs.append(dict(held_source=held,target=target,model=model,parameter=val,inner_mae=scores,train_donors=donors[tr].tolist(),test_donors=donors[te].tolist(),inner_splits=[dict(train=donors[it].tolist(),validation=donors[iv].tolist()) for it,iv in inner],selected_genes=[dict(cell=c,ids=a.gene_ids.iloc[ii].tolist(),symbols=a['index'].iloc[ii].tolist()) for c,a,ii in zip(['Astrocyte','Micro-PVM'],ann,pack[-1])]))
  print('held source',held,'done',flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'recruitment_predictions.csv',index=False);pd.DataFrame(metrics).to_csv(ROOT/'recruitment_metrics.csv',index=False);(ROOT/'recruitment_training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8');print(pd.DataFrame(metrics).round(4).to_string(index=False))
if __name__=='__main__':main()
