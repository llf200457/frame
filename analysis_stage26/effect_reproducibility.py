"""Cross-cohort module contrasts and gene-effect concordance.

Effect directions use AD minus control within each cohort. Analyses are
descriptive after the added datasets were identified; no features are selected.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE if (BASE/'modules').is_dir() else BASE.parents[1]
OUT=ROOT/'analysis_stage26'
SEED=20260929
M=pd.read_csv(ROOT/'modules/frozen_pca_modules.csv')
P=pd.read_csv(OUT/'external_transport_predictions.csv')
assert set(P.cohort)=={'GSE63060','GSE63061','GSE140829','GSE97760'}

def hedges_g(a,b):
    na=len(a);nb=len(b)
    pooled=np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return (1-3/(4*(na+nb)-9))*(np.mean(a)-np.mean(b))/pooled if pooled>0 else np.nan

rows=[]
for name,frame in P.groupby('cohort'):
    frame=frame[frame.status.isin(['AD','CTL'])]
    for k in sorted(M.module.unique()):
        a=frame.loc[frame.status.eq('AD'),k].to_numpy()
        b=frame.loc[frame.status.eq('CTL'),k].to_numpy()
        point=hedges_g(a,b)
        rng=np.random.default_rng(SEED)
        draws=[hedges_g(rng.choice(a,len(a),replace=True),rng.choice(b,len(b),replace=True)) for _ in range(2000)]
        lo,hi=np.nanquantile(draws,[.025,.975])
        rows.append({'cohort':name,'module':k,'AD_n':len(a),'CTL_n':len(b),
                     'g':point,'g_low':lo,'g_high':hi,
                     'AD_mean_score':float(np.mean(a)),'CTL_mean_score':float(np.mean(b))})
pd.DataFrame(rows).to_csv(OUT/'module_effects.csv',index=False)

discovery=pd.read_pickle(ROOT/'processed/GSE63060_gene_expression.pkl',compression='gzip')
first=pd.read_pickle(ROOT/'processed/GSE63061_gene_expression.pkl',compression='gzip')
second=pd.read_csv(ROOT/'raw/GSE140829/GSE140829_final_normalized_data.txt.gz',sep='\t',index_col=0)
smallraw=pd.read_csv(ROOT/'raw/GSE97760/GSE97760_loess.txt.gz',sep='\t').dropna(subset=['GeneSymbol'])
smallraw=smallraw[smallraw.GeneSymbol.str.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*')]
small=np.log2(smallraw[[c for c in smallraw if c.startswith('Norm_')]].astype(float)+1)
small.index=smallraw.GeneSymbol
small=small.groupby(level=0).mean()
data={'GSE63060':discovery,'GSE63061':first,'GSE140829':second,'GSE97760':small}
effects={}
for name,x in data.items():
    lab=P.loc[P.cohort.eq(name),['sample','status']].set_index('sample').status
    a=x.loc[x.index.intersection(M.gene),lab[lab.eq('AD')].index].to_numpy()
    b=x.loc[x.index.intersection(M.gene),lab[lab.eq('CTL')].index].to_numpy()
    ga=np.nanmean(a,axis=1);gb=np.nanmean(b,axis=1)
    va=np.nanvar(a,axis=1,ddof=1);vb=np.nanvar(b,axis=1,ddof=1)
    na=np.isfinite(a).sum(axis=1);nb=np.isfinite(b).sum(axis=1)
    pooled=np.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2))
    g=(1-3/(4*(na+nb)-9))*(ga-gb)/pooled
    effects[name]=pd.Series(g,index=x.index.intersection(M.gene))

pd.DataFrame(effects).rename_axis('gene').to_csv(OUT/'gene_effects.csv')
correlations=[]
for j,left in enumerate(data):
    for right in list(data)[j+1:]:
        pair=pd.concat([effects[left],effects[right]],axis=1,keys=['left','right']).dropna()
        for module,part in [('all',pair)]+[(k,pair.loc[pair.index.intersection(M.loc[M.module.eq(k),'gene'])]) for k in ['M01','M07']]:
            correlations.append({'cohort_a':left,'cohort_b':right,'module':module,
                                 'common_genes':len(part),'spearman_rho':spearmanr(part.left,part.right).statistic,
                                 'same_direction_fraction':float((np.sign(part.left)==np.sign(part.right)).mean())})
pd.DataFrame(correlations).to_csv(OUT/'gene_effect_concordance.csv',index=False)
print(pd.DataFrame(rows).query("module in ['M01','M07']").to_string(index=False,float_format=lambda x:f'{x:.3f}'))
print(pd.DataFrame(correlations).to_string(index=False,float_format=lambda x:f'{x:.3f}'))
