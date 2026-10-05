from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata
ROOT=Path(__file__).resolve().parent
def rank_corr(x,y):
 x=rankdata(x,axis=1);y=rankdata(y,axis=1);x=x-x.mean(axis=1,keepdims=True);y=y-y.mean(axis=1,keepdims=True);den=np.sqrt((x*x).sum(axis=1)*(y*y).sum(axis=1));return np.divide((x*y).sum(axis=1),den,out=np.full(len(x),np.nan),where=den>0)
def main():
 p=pd.read_csv(ROOT/'external_scores.csv');out=[];sources=[];rng=np.random.default_rng(41001)
 for (target,component,margin),z in p.groupby(['target','component','marker_margin']):
  score=z.pivot(index='donor_id',columns='region',values='frozen_RNA_score');tau=z.pivot(index='donor_id',columns='region',values='ptau_percent');assert len(score)==12 and score[['EC','SSC']].notna().all().all();tau=np.log1p(tau)
  for comparison in ['EC','SSC','EC_minus_SSC']:
   x=score[comparison].to_numpy() if comparison!='EC_minus_SSC' else (score.EC-score.SSC).to_numpy();y=tau[comparison].to_numpy() if comparison!='EC_minus_SSC' else (tau.EC-tau.SSC).to_numpy();rho=spearmanr(x,y);ix=rng.integers(len(x),size=(4000,len(x)));boot=rank_corr(x[ix],y[ix]);ci=np.nanquantile(boot,[.025,.975]);perm=np.argsort(rng.random((19999,len(x))),axis=1);null=rank_corr(np.broadcast_to(x,perm.shape),y[perm]);pv=(1+(abs(null)>=abs(rho.statistic)-1e-12).sum())/20000
   out.append(dict(target=target,component=component,marker_margin=float(margin),comparison=comparison,n_donors=len(x),spearman=float(rho.statistic),ci_low=float(ci[0]),ci_high=float(ci[1]),permutation_p=float(pv),asymptotic_p=float(rho.pvalue),bootstrap_valid=int(np.isfinite(boot).sum())))
   sources.extend(dict(target=target,component=component,marker_margin=float(margin),comparison=comparison,donor_id=d,RNA_score=float(v),log1p_ptau=float(t)) for d,v,t in zip(score.index,x,y))
 a=pd.DataFrame(out);a['bh_q']=np.nan
 for margin in [.25,.50]:
  idx=a.index[a.marker_margin==margin];order=idx[np.argsort(a.loc[idx,'permutation_p'])];q=np.minimum.accumulate((a.loc[order,'permutation_p'].to_numpy()*len(order)/np.arange(1,len(order)+1))[::-1])[::-1];a.loc[order,'bh_q']=np.minimum(q,1)
 a.to_csv(ROOT/'external_associations.csv',index=False);pd.DataFrame(sources).to_csv(ROOT/'external_association_source.csv',index=False)
 counts=pd.read_csv(ROOT/'cell_qc_counts.csv');q=counts[(counts.marker_margin==.25)&counts.cell.isin(['Astrocyte','Micro-PVM'])];assert q.groupby('gsm').size().eq(2).all() and q.nuclei.min()>=20;assert len(q)==48
 identity=[]
 # Gene ID alignment audit: versions stripped with PAR_Y suffix retained, no symbol substitution.
 import joblib
 for target in ['Tau_panel','Tau_DFC']:
  b=joblib.load(ROOT/('frozen_'+target+'.joblib'));sel=[(c['cell'],g,s) for c in b['selected_genes'] for g,s in zip(c['gene_ids'],c['symbols'])]
  for f in sorted((ROOT/'pseudobulk').glob('*.npz')):
   v=np.load(f);ids=v['gene_ids'].tolist();sy=v['symbols'].tolist();mapping=dict(zip(ids,sy))
   for cell,g,s in sel:identity.append(dict(gsm=f.stem,target=target,cell=cell,gene_id=g,source_symbol=s,external_symbol=mapping[g],symbol_identical=s==mapping[g]))
 pd.DataFrame(identity).to_csv(ROOT/'gene_identity_audit.csv',index=False)
 (ROOT/'external_audit.json').write_text(json.dumps({'external_people':12,'samples':24,'QC_min_glial_nuclei':int(q.nuclei.min()),'all_20_gene_inputs_mapped_each_sample':True,'frozen_score_not_full_predictor':True,'primary_BH_family':12,'sensitivity_BH_family':12,'bootstrap_draws':4000,'permutation_draws':19999,'cell_annotations':'operational canonical-marker labels, original annotations unavailable','pathology_mismatch':'percentage-positive-cells versus percentage-positive-area, no calibration claims'},indent=2),encoding='utf-8');print(a.round(4).to_string(index=False))
if __name__=='__main__':main()
