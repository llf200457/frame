from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr,t as student_t
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parent
def ccc(x,y):return float(2*np.mean((x-x.mean())*(y-y.mean()))/(x.var()+y.var()+(x.mean()-y.mean())**2))
def summarize():
 p=pd.read_csv(ROOT/'predictions.csv');rows=[]
 for (t,m,s),z in p.groupby(['target','model','seed']):
  y=z.observed.to_numpy();v=z.predicted.to_numpy();rows.append(dict(target=t,model=m,seed=int(s),n=len(z),mae=float(abs(y-v).mean()),r2=float(r2_score(y,v)),ccc=ccc(y,v),bias=float((v-y).mean())))
 pd.DataFrame(rows).to_csv(ROOT/'metrics.csv',index=False);rng=np.random.default_rng(39001);comparisons=[]
 intervals=[];irng=np.random.default_rng(39004)
 for (target,seed),z in p[p.model=='genes'].groupby(['target','seed']):
  y=z.observed.to_numpy();v=z.predicted.to_numpy();ix=irng.integers(0,len(z),(4000,len(z)));yy=y[ix];vv=v[ix];r2=1-((yy-vv)**2).sum(axis=1)/((yy-yy.mean(axis=1,keepdims=True))**2).sum(axis=1);lo,hi=np.quantile(r2,[.025,.975]);intervals.append(dict(target=target,seed=int(seed),r2_low=float(lo),r2_high=float(hi)))
 pd.DataFrame(intervals).to_csv(ROOT/'r2_intervals.csv',index=False)
 for target,z in p.groupby('target'):
  z=z.assign(error=abs(z.observed-z.predicted));e=z.groupby(['donor_id','model']).error.mean().unstack()
  for base in ['metadata','quality']:
   for model in ['genes','markers','orthogonal']:
    d=(e[base]-e[model]).to_numpy();draw=d[rng.integers(0,len(d),(4000,len(d)))].mean(axis=1)
    comparisons.append(dict(target=target,baseline=base,model=model,n=len(d),improvement=float(d.mean()),ci_low=float(np.quantile(draw,.025)),ci_high=float(np.quantile(draw,.975)),relative_improvement=float(d.mean()/e[base].mean())))
 pd.DataFrame(comparisons).to_csv(ROOT/'paired_bootstrap.csv',index=False)
 lum=pd.read_csv(ROOT.parent/'analysis_stage32_marker_pivot/SEA_AD_Luminex_MTG.csv').set_index('Donor ID');cohort=pd.read_csv(ROOT/'analysis_cohort.csv').set_index('donor_id');assocs=[];sources=[]
 for (target,model),z in p.groupby(['target','model']):
  z=z.groupby('donor_id')[['observed','predicted']].mean().join(lum).join(cohort.iloc[:,:6]);pred=z.predicted.to_numpy()
  cov=z.iloc[:,-6:].to_numpy(float);cov=np.where(np.isfinite(cov),cov,np.nanmedian(cov,axis=0));cov=np.column_stack([np.ones(len(z)),cov])
  for col in lum.columns:
   v=np.log1p(pd.to_numeric(z[col],errors='coerce').to_numpy());valid=np.isfinite(v);x=pred[valid];y=v[valid];c=cov[valid];rx=x-c@np.linalg.lstsq(c,x,rcond=None)[0];ry=y-c@np.linalg.lstsq(c,y,rcond=None)[0];pc=pearsonr(rx,ry)
   df=len(x)-np.linalg.matrix_rank(c)-1;r=float(pc.statistic);pv=float(2*student_t.sf(abs(r)*np.sqrt(df/max(1-r*r,1e-15)),df))
   rho=spearmanr(x,y);assocs.append(dict(target=target,model=model,measurement=col,n=len(x),spearman=float(rho.statistic),spearman_p=float(rho.pvalue),partial_r=r,partial_p=pv,partial_df=int(df)))
   if model=='genes' and target in ['Tau_panel','Tau_DFC']:
    sources.extend(dict(donor_id=d,target=target,measurement=col,predicted=float(a),log1p_luminex=float(b),residual_prediction=float(c),residual_luminex=float(e)) for d,a,b,c,e in zip(z.index[valid],x,y,rx,ry))
 a=pd.DataFrame(assocs)
 # BH across the full 4-target x 5-model x 8-measurement exploratory family.
 for col in ['spearman_p','partial_p']:
  order=np.argsort(a[col].to_numpy());ps=a[col].to_numpy()[order];adj=np.minimum.accumulate((ps*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1];vals=np.empty(len(ps));vals[order]=np.minimum(adj,1);a[col+'_bh']=vals
 a.to_csv(ROOT/'orthogonal_measurement_associations.csv',index=False);pd.DataFrame(sources).to_csv(ROOT/'orthogonal_measurement_source.csv',index=False)
 controls=[]
 for (target,model),z in p.groupby(['target','model']):
  z=z.groupby('donor_id')[['observed','predicted']].mean().join(lum).join(cohort.iloc[:,:30]);base=z.iloc[:,-30:].to_numpy(float);base=np.where(np.isfinite(base),base,np.nanmedian(base,axis=0))
  for buffer in ['RIPA','GuHCl']:
   pc=buffer+'_pTAU_pg_per_ug'; extra=[buffer+'_'+q+'_pg_per_ug' for q in ['tTAU','ABeta40','ABeta42']];x=z.predicted.to_numpy();y=np.log1p(z[pc].to_numpy(float));cc=np.column_stack([np.ones(len(z)),base,np.log1p(z[extra].to_numpy(float))]);valid=np.isfinite(cc).all(axis=1)&np.isfinite(y);x=x[valid];y=y[valid];cc=cc[valid];rx=x-cc@np.linalg.lstsq(cc,x,rcond=None)[0];ry=y-cc@np.linalg.lstsq(cc,y,rcond=None)[0];r=float(pearsonr(rx,ry).statistic);df=len(x)-np.linalg.matrix_rank(cc)-1
   controls.append(dict(target=target,model=model,buffer=buffer,n=len(x),r=r,df=int(df),p=float(2*student_t.sf(abs(r)*np.sqrt(df/max(1-r*r,1e-15)),df))))
 b=pd.DataFrame(controls);order=np.argsort(b.p.to_numpy());ps=b.p.to_numpy()[order];q=np.minimum.accumulate((ps*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1];vv=np.empty(len(ps));vv[order]=np.minimum(q,1);b['q']=vv;b.to_csv(ROOT/'tau_specificity_full_adjustment.csv',index=False)
 print(pd.DataFrame(rows).groupby(['target','model'])[['mae','r2','ccc']].mean().round(4).to_string());print(pd.DataFrame(comparisons).query("model=='genes'").round(4).to_string(index=False));print(a.query("model=='genes' and target=='Tau_panel'").round(4).to_string(index=False))
if __name__=='__main__':summarize()
