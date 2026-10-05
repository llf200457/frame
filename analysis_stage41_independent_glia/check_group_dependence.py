"""Post-result diagnosis-stratified diagnostic, not part of the locked 12 tests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
ROOT=Path(__file__).resolve().parent

def main():
 p=pd.read_csv(ROOT/'external_association_source.csv')
 m=pd.read_csv(ROOT/'external_metadata.csv').drop_duplicates('donor_id').set_index('donor_id')
 rng=np.random.default_rng(41002);rows=[];loo=[];within=[]
 for target in ['Tau_panel','Tau_DFC']:
  for region in ['EC','SSC']:
   z=p[(p.target==target)&(p.component=='all')&(p.marker_margin==.25)&(p.comparison==region)].copy()
   group=(m.loc[z.donor_id,'disease state'].to_numpy()=='AD').astype(float)
   x=rankdata(z.RNA_score);y=rankdata(z.log1p_ptau);C=np.column_stack([np.ones(len(x)),group]);xr=x-C@np.linalg.lstsq(C,x,rcond=None)[0];yr=y-C@np.linalg.lstsq(C,y,rcond=None)[0]
   rho=float(np.corrcoef(xr,yr)[0,1]);perms=np.tile(np.arange(len(x)),(19999,1))
   for g in [0,1]:
    ix=np.where(group==g)[0];perms[:,ix]=ix[np.argsort(rng.random((19999,len(ix))),axis=1)]
   null=yr[perms]@xr/np.sqrt((yr*yr).sum()*(xr*xr).sum());pv=float((1+(abs(null)>=abs(rho)-1e-12).sum())/20000)
   rows.append(dict(target=target,region=region,n_donors=len(x),diagnosis_adjusted_rank_correlation=rho,within_diagnosis_permutation_p=pv))
   for g in [0,1]:
    q=z.iloc[np.where(group==g)[0]];s=spearmanr(q.RNA_score,q.log1p_ptau);within.append(dict(target=target,region=region,group='AD' if g else 'Control',n_donors=len(q),spearman=float(s.statistic)))
   for d in z.donor_id:
    q=z[z.donor_id!=d];loo.append(dict(target=target,region=region,omitted_donor=d,n_donors=len(q),spearman=float(spearmanr(q.RNA_score,q.log1p_ptau).statistic)))
 a=pd.DataFrame(rows);order=np.argsort(a.within_diagnosis_permutation_p);q=np.minimum.accumulate((a.within_diagnosis_permutation_p.to_numpy()[order]*4/np.arange(1,5))[::-1])[::-1];a['bh_q']=np.nan;a.loc[order,'bh_q']=np.minimum(q,1)
 a.to_csv(ROOT/'diagnosis_adjusted_diagnostics.csv',index=False);pd.DataFrame(within).to_csv(ROOT/'within_group_correlations.csv',index=False);pd.DataFrame(loo).to_csv(ROOT/'leave_one_donor_diagnostics.csv',index=False)
 print(a.round(4).to_string(index=False));print(pd.DataFrame(within).round(4).to_string(index=False));print(pd.DataFrame(loo).groupby(['target','region']).spearman.agg(['min','max']).round(4))
if __name__=='__main__':main()
