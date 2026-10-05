from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parent
def ccc(y,p):return float(2*np.mean((y-y.mean())*(p-p.mean()))/(y.var()+p.var()+(y.mean()-p.mean())**2))
def main():
 p=pd.read_csv(ROOT/'predictions.csv');old=pd.read_csv(ROOT.parent/'analysis_stage39_tau_panel/predictions.csv');old=old[old.model.isin(['genes','quality','metadata'])&old.target.isin(p.target.unique())];p=pd.concat([p,old],ignore_index=True);metrics=[];diff=[];rng=np.random.default_rng(40001)
 for (target,model,seed),z in p.groupby(['target','model','seed']):
  y=z.observed.to_numpy();v=z.predicted.to_numpy();metrics.append(dict(target=target,model=model,seed=seed,n=len(z),mae=float(abs(y-v).mean()),r2=float(r2_score(y,v)),ccc=ccc(y,v),spearman=float(spearmanr(y,v).statistic),bias=float((v-y).mean())))
 m=pd.DataFrame(metrics);m.to_csv(ROOT/'metrics.csv',index=False);m.groupby(['target','model'])[['mae','r2','ccc','spearman']].mean().to_csv(ROOT/'summary.csv')
 for target,z in p.groupby('target'):
  e=z.assign(error=abs(z.observed-z.predicted)).groupby(['donor_id','model']).error.mean().unstack()
  models=sorted(set(z.model)-{'genes','quality','metadata'})
  pairs=[(model,base) for model in models for base in ['genes','quality']]+[('interaction_kernel','additive_kernel'),('additive_kernel','astro_kernel'),('additive_kernel','micro_kernel')]
  for model,base in pairs:
   d=(e[base]-e[model]).to_numpy();draw=d[rng.integers(len(d),size=(4000,len(d)))].mean(axis=1);sign=rng.choice([-1,1],size=(19999,len(d)));stat=(sign*d).mean(axis=1);pv=float((1+(abs(stat)>=abs(d.mean())).sum())/20000)
   mm=m[(m.target==target)&m.model.isin([model,base])].pivot(index='seed',columns='model',values='mae');diff.append(dict(target=target,model=model,baseline=base,improvement=float(d.mean()),ci_low=float(np.quantile(draw,.025)),ci_high=float(np.quantile(draw,.975)),signflip_p=pv,improved_seeds=int((mm[model]<mm[base]).sum()),n=len(d)))
 d=pd.DataFrame(diff);d['holm_p_vs_genes']=np.nan;idx=d.index[d.baseline=='genes'];order=idx[np.argsort(d.loc[idx,'signflip_p'])];adjust=np.maximum.accumulate(d.loc[order,'signflip_p'].to_numpy()*np.arange(len(order),0,-1));d.loc[order,'holm_p_vs_genes']=np.minimum(adjust,1);d.to_csv(ROOT/'paired_comparisons.csv',index=False)
 logs=json.loads((ROOT/'training_log.json').read_text());assert len(logs)==405 and len(p[p.model.isin(models)])==6156
 for row in logs:
  assert not set(row['train_donors'])&set(row['test_donors'])
  for inn in row['inner_splits']:
   assert not set(inn['train'])&set(inn['validation']);assert set(inn['train'])|set(inn['validation'])<=set(row['train_donors'])
 assert (p.groupby(['target','model','seed']).size()==76).all();assert not p.duplicated(['donor_id','target','model','seed']).any()
 (ROOT/'audit.json').write_text(json.dumps(dict(outer_fits=405,new_oof_predictions=6156,all_groups_76_unique_donors=True,outer_and_inner_donor_disjoint=True,bootstrap_draws=4000,signflip_draws=19999,holm_family_size=27,statistical_scope='Exploratory; paired tests condition on dependent OOF fits. No independent confirmation or selection-adjusted validation.'),indent=2),encoding='utf-8')
 print(m.groupby(['target','model'])[['mae','r2','ccc']].mean().round(4).to_string());print(d[(d.baseline=='genes')|((d.model=='interaction_kernel')&(d.baseline=='additive_kernel'))].round(4).to_string(index=False))
if __name__=='__main__':main()
