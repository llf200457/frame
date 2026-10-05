from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parent
old=pd.read_csv(ROOT.parent/'analysis_stage36_stability_calibration/combined_predictions.csv')
target='percent AT8 positive area'
old=old[(old.target==target)&old.seed.isin([2026,2027])&old.model.isin(['metadata','pooled'])]
new=pd.read_csv(ROOT/'predictions.csv'); df=pd.concat([old,new],ignore_index=True)
keys=['seed','donor_id','region']; assert not df.duplicated(keys+['model']).any()
wide=df.pivot(index=keys,columns='model',values='predicted'); assert wide.notna().all().all() and len(wide)==994
truth=df.groupby(keys)['observed'].agg(['min','max']); assert np.allclose(truth['min'],truth['max'])
df.to_csv(ROOT/'combined_predictions.csv',index=False)
metrics=[]
for (seed,r,m),s in df.groupby(['seed','region','model']):
    metrics.append(dict(seed=seed,region=r,model=m,n=len(s),mae=np.mean(abs(s.predicted-s.observed)),r2=r2_score(s.observed,s.predicted),bias=np.mean(s.predicted-s.observed)))
metrics=pd.DataFrame(metrics); metrics.to_csv(ROOT/'region_metrics.csv',index=False)
summary=metrics.groupby(['seed','model']).agg(macro_mae=('mae','mean'),macro_r2=('r2','mean'),positive_r2_regions=('r2',lambda a:int((a>0).sum()))).reset_index(); summary.to_csv(ROOT/'macro_summary.csv',index=False)
ids=np.array(sorted(df.donor_id.unique())); did=pd.Categorical(wide.index.get_level_values('donor_id'),categories=ids).codes
group=pd.factorize(pd.MultiIndex.from_arrays([wide.index.get_level_values('seed'),wide.index.get_level_values('region')]))[0]
y=truth['min'].reindex(wide.index).to_numpy(); errs=abs(wide.to_numpy()-y[:,None]); models=list(wide.columns)
def macro_error(w):
    return np.mean([np.average(errs[group==g],axis=0,weights=w[group==g]) for g in np.unique(group) if w[group==g].sum()>0],axis=0)
pairs=[('metadata','quality_only'),('pooled','quality_gene'),('quality_only','quality_gene'),('metadata','quality_gene')]
base=macro_error(np.ones(len(wide))); rng=np.random.default_rng(3701); draws=[]
for i in range(2000):
    count=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids)); draws.append(macro_error(count[did]))
draws=np.array(draws); intervals=[]
for a,b in pairs:
    delta=draws[:,models.index(a)]-draws[:,models.index(b)]; lo,hi=np.quantile(delta,[.025,.975]); intervals.append(dict(reference=a,candidate=b,mae_improvement=base[models.index(a)]-base[models.index(b)],conditional_low=lo,conditional_high=hi))
pd.DataFrame(intervals).to_csv(ROOT/'paired_intervals.csv',index=False)

# Descriptive residual association; not used by any predictive model.
allold=pd.read_csv(ROOT.parent/'analysis_stage36_stability_calibration/combined_predictions.csv'); allold=allold[(allold.target==target)&allold.model.isin(['metadata','pooled'])]
a=allold.groupby(['donor_id','region','model']).agg(observed=('observed','first'),predicted=('predicted','mean')).reset_index()
tw=a.pivot(index=['donor_id','region'],columns='model',values='predicted'); obs=a.groupby(['donor_id','region'])['observed'].first().reindex(tw.index).to_numpy()
index=tw.index.to_frame(index=False); matrix=np.column_stack([obs,tw['metadata'].to_numpy(),tw['pooled'].to_numpy()])
designs={'donor_only':np.column_stack([np.ones(len(index)),pd.get_dummies(index.donor_id,drop_first=True,dtype=float).to_numpy()]),'donor_and_region':np.column_stack([np.ones(len(index)),pd.get_dummies(index.donor_id,drop_first=True,dtype=float).to_numpy(),pd.get_dummies(index.region,drop_first=True,dtype=float).to_numpy()])}
dids=pd.Categorical(index.donor_id,categories=ids).codes; resout=index.copy(); associations=[]
for mode,z in designs.items():
    residual=matrix-z@np.linalg.lstsq(z,matrix,rcond=None)[0]
    for k,name in enumerate(['observed','metadata','pooled']): resout[mode+'_'+name]=residual[:,k]
    vals=[]; rng=np.random.default_rng(3702)
    for i in range(2000):
        count=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids)); rows=np.repeat(np.arange(len(index)),count[dids]); zb=z[rows]; mb=matrix[rows]; rb=mb-zb@np.linalg.lstsq(zb,mb,rcond=None)[0]
        vals.append([pearsonr(rb[:,0],rb[:,k]).statistic for k in [1,2]])
    for j,name in enumerate(['metadata','pooled']):
        lo,hi=np.quantile(np.array(vals)[:,j],[.025,.975]); associations.append(dict(adjustment=mode,model=name,n_records=len(index),donors=len(ids),pearson=pearsonr(residual[:,0],residual[:,j+1]).statistic,spearman=spearmanr(residual[:,0],residual[:,j+1]).statistic,descriptive_cluster_low=lo,descriptive_cluster_high=hi))
resout.to_csv(ROOT/'residual_records.csv',index=False); pd.DataFrame(associations).to_csv(ROOT/'residual_associations.csv',index=False)
logs=json.loads((ROOT/'training_log.json').read_text(encoding='utf-8')); ilogs=json.loads((ROOT/'inner_splits.json').read_text(encoding='utf-8'))
for s in logs: assert not set(s['train_donors'])&set(s['test_donors']) and s['held'] not in s['train_regions']
for s in ilogs: assert not set(s['train_donors'])&set(s['valid_donors']) and not set(s['train_regions'])&set(s['valid_regions'])
files=[ROOT.parent/'analysis_stage33_seaad_pairing'/f for f in ['Astrocyte_official_observations.csv','Immune_official_observations.csv','Astrocyte_donor_region_counts.npz','Immune_donor_region_counts.npz','Astrocyte_paired_measured_pathology.csv','Immune_paired_measured_pathology.csv','SEA_AD_MTG_baseline_covariates.csv']]
(ROOT/'validation_audit.json').write_text(json.dumps(dict(people=84,records=497,new_fits=len(logs),inner_definitions=len(ilogs),outer_and_inner_disjoint=True,new_predictions=len(new),combined_predictions=len(df),bootstrap_repetitions=2000,conditional_intervals=True,source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2),encoding='utf-8')
print(summary.to_string(index=False)); print(pd.DataFrame(intervals).to_string(index=False)); print(pd.DataFrame(associations).to_string(index=False))
