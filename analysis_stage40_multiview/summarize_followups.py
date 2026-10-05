from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from summarize import ccc
ROOT=Path(__file__).resolve().parent
def holm(ps):
 order=np.argsort(ps);adj=np.maximum.accumulate(ps[order]*np.arange(len(ps),0,-1));out=np.empty(len(ps));out[order]=np.minimum(adj,1);return out
def main():
 new=pd.read_csv(ROOT/'predictions.csv');new=new[new.model=='random_forest'];q=pd.read_csv(ROOT/'quality_rf_predictions.csv');a=pd.read_csv(ROOT/'rf_ablation_predictions.csv');old=pd.read_csv(ROOT.parent/'analysis_stage39_tau_panel/predictions.csv');old=old[(old.model=='genes')&old.target.isin(new.target.unique())];p=pd.concat([new,q,a,old],ignore_index=True);metrics=[];contrasts=[];rng=np.random.default_rng(40004)
 for (target,model,seed),z in p.groupby(['target','model','seed']):metrics.append(dict(target=target,model=model,seed=int(seed),n=len(z),mae=float(abs(z.observed-z.predicted).mean()),r2=float(r2_score(z.observed,z.predicted)),ccc=ccc(z.observed.to_numpy(),z.predicted.to_numpy())))
 m=pd.DataFrame(metrics);m.to_csv(ROOT/'followup_metrics.csv',index=False)
 for target,z in p.groupby('target'):
  e=z.assign(error=abs(z.observed-z.predicted)).groupby(['donor_id','model']).error.mean().unstack()
  for base in ['quality_rf','metadata_rf','rf_astro20','rf_micro20','rf_no_mt']:
   d=(e[base]-e.random_forest).to_numpy();draw=d[rng.integers(len(d),size=(4000,len(d)))].mean(axis=1);sign=rng.choice([-1,1],size=(19999,len(d)));pv=(1+(abs((sign*d).mean(axis=1))>=abs(d.mean())).sum())/20000;mm=m[(m.target==target)&m.model.isin([base,'random_forest'])].pivot(index='seed',columns='model',values='mae');contrasts.append(dict(target=target,baseline=base,improvement=float(d.mean()),ci_low=float(np.quantile(draw,.025)),ci_high=float(np.quantile(draw,.975)),signflip_p=float(pv),improved_seeds=int((mm[base]>mm.random_forest).sum())))
 c=pd.DataFrame(contrasts);c['holm_p']=np.nan;c['holm_family']=''
 for baselist,family in [(['quality_rf','metadata_rf'],'six quality/metadata contrasts'),(['rf_astro20','rf_micro20','rf_no_mt'],'nine RF component contrasts')]:
  mask=c.baseline.isin(baselist);c.loc[mask,'holm_p']=holm(c.loc[mask,'signflip_p'].to_numpy());c.loc[mask,'holm_family']=family
 c.to_csv(ROOT/'followup_comparisons.csv',index=False)
 # Post hoc fixed burden strata; average loss across runs without averaging predictions.
 z=p[p.target=='Tau_DFC'].copy();z['error']=abs(z.observed-z.predicted);z['stratum']=pd.cut(np.expm1(z.observed),[-np.inf,.1,1,np.inf],right=False,labels=['<0.1%','0.1 to <1%','>=1%']);strata=z.groupby(['model','stratum'],observed=True).agg(n_donors=('donor_id','nunique'),mae=('error','mean')).reset_index();strata.to_csv(ROOT/'DFC_burden_strata.csv',index=False)
 highest=z.groupby('donor_id')['observed'].mean().idxmax();deletion=[]
 for (model,seed),s in z.groupby(['model','seed']):
  ss=s[s.donor_id!=highest];deletion.append(dict(model=model,seed=int(seed),excluded_donor=highest,full_r2=float(r2_score(s.observed,s.predicted)),without_highest_r2=float(r2_score(ss.observed,ss.predicted)),without_highest_mae=float(abs(ss.observed-ss.predicted).mean()),n=len(ss)))
 pd.DataFrame(deletion).to_csv(ROOT/'DFC_highest_burden_deletion.csv',index=False)
 # Count all completed log blocks and verify donor separation.
 audit={}
 for file,expected in [('training_log.json',405),('quality_rf_training_log.json',90),('recruitment_training_log.json',24),('rf_ablation_training_log.json',135)]:
  logs=json.loads((ROOT/file).read_text());assert len(logs)==expected
  for r in logs:
   assert not set(r['train_donors'])&set(r['test_donors'])
   for inn in r['inner_splits']:assert not set(inn['train'])&set(inn['validation']) and (set(inn['train'])|set(inn['validation']))<=set(r['train_donors'])
  audit[file]=len(logs)
 for (_,_,_),s in p.groupby(['target','model','seed']):assert len(s)==76 and s.donor_id.nunique()==76
 hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')};(ROOT/'followup_audit.json').write_text(json.dumps({'outer_fits':audit,'all_splits_disjoint':True,'scripts':hashes,'highest_burden_donor':highest,'frozen_budget_complete':True},indent=2),encoding='utf-8');print(m.groupby(['target','model'])[['mae','r2','ccc']].mean().round(4).to_string());print(c.round(4).to_string(index=False));print(pd.DataFrame(deletion).groupby('model')[['without_highest_r2']].mean().round(4).to_string())
if __name__=='__main__':main()
