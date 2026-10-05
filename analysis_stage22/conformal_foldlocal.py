"""Fit-only feature construction for valid discovery calibration separation."""
from nested_validation import *

t,e=load();tl=t[t.label_formal.isin(['AD','Control'])];ev=e[e.status.isin(['AD','CTL'])]
g1=pd.read_pickle(ROOT/'processed/GSE63060_gene_expression.pkl',compression='gzip');g2=pd.read_pickle(ROOT/'processed/GSE63061_gene_expression.pkl',compression='gzip')
genes=sorted(set(g1.index)&set(g2.index));a=g1.loc[genes,tl.index].T.to_numpy();ae=g2.loc[genes,e.index].T.to_numpy();y=tl.label_formal.eq('AD').astype(int).to_numpy();ye=ev.status.eq('AD').astype(int).to_numpy();index=e.index.get_indexer(ev.index)
rows=[]
for frac in [.1,.2,.3,.4]:
 for seed in range(2025,2045):
  fi,ca=train_test_split(np.arange(len(y)),test_size=frac,stratify=y,random_state=seed)
  xt,xc,xx=represent(a[fi],[a[ca],ae])[8];m=model(seed).fit(xt,y[fi]);pc=m.predict_proba(xc)[:,1];pe=m.predict_proba(xx[index])[:,1]
  nc=np.where(y[ca],1-pc,pc);rank=int(np.ceil((len(ca)+1)*.9));q=np.sort(nc)[rank-1]
  domain=make_pipeline(StandardScaler(),LogisticRegression(max_iter=10000,random_state=seed)).fit(np.r_[xt,xx],np.r_[np.zeros(len(fi)),np.ones(len(xx))])
  dp=domain.predict_proba(xc)[:,1];raw=dp/np.clip(1-dp,1e-6,None)*len(fi)/len(xx);w=np.clip(raw,.05,20);oo=np.argsort(nc);qw=nc[oo][np.searchsorted(np.cumsum(w[oo])/w.sum(),.9)]
  for name,qq in [('Ordinary',q),('Reweighted',qw)]:
   c0=pe<=qq;c1=1-pe<=qq;size=c0.astype(int)+c1.astype(int)
   rows.append({'calibration_fraction':frac,'calibration_n':len(ca),'seed':seed,'method':name,'q':qq,'coverage':np.where(ye,c1,c0).mean(),'AD_coverage':c1[ye==1].mean(),'CTL_coverage':c0[ye==0].mean(),'empty_rate':np.mean(size==0),'both_rate':np.mean(size==2),'mean_set_size':size.mean(),'effective_n':w.sum()**2/(w*w).sum(),'weight_clip_low_fraction':np.mean(raw<.05),'weight_clip_high_fraction':np.mean(raw>20)})
 print('fit-only conformal',frac,flush=True)
pd.DataFrame(rows).to_csv(OUT/'conformal_fit_only_sensitivity.csv',index=False)
print(pd.DataFrame(rows).groupby(['calibration_n','method'])[['coverage','AD_coverage','CTL_coverage']].mean())
