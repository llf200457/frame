"""Verify and score a second independent RNA-seq cohort without target fitting."""
from pathlib import Path
import csv, gzip, hashlib, json, re
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis_stage28'
matrix_path = OUT / 'GSE248417_ADvsControl.mRNA.gene_level.deg.txt.gz'
series_path = OUT / 'GSE248417_series_matrix.txt.gz'
assert matrix_path.is_file() and series_path.is_file()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

with gzip.open(series_path,'rt',encoding='utf-8') as f:
    rows = {r[0]:r[1:] for line in f if line.startswith('!Sample_')
            for r in [next(csv.reader([line],delimiter='\t'))]}
ids = rows['!Sample_description']
geo = rows['!Sample_geo_accession']
titles = rows['!Sample_title']
with gzip.open(series_path,'rt',encoding='utf-8') as f:
    disease_rows = [next(csv.reader([line],delimiter='\t'))[1:] for line in f
                    if line.startswith('!Sample_characteristics_ch1')
                    and 'disease state:' in line]
assert len(disease_rows)==1
status = [v.split(':',1)[1].strip() for v in disease_rows[0]]
meta = pd.DataFrame({'sample':ids,'gsm':geo,'title':titles,'status':status})
assert len(meta)==98 and meta['sample'].is_unique
assert meta.status.value_counts().to_dict()=={'AD':49,'Control':49}
assert all((a.startswith('AD') if s=='AD' else a.startswith('C'))
           for a,s in zip(meta['sample'],meta.status))

with gzip.open(matrix_path,'rt',encoding='utf-8') as f:
    header = next(csv.reader(f,delimiter='\t'))
sample_cols = [c for c in header if c in set(meta['sample'])]
assert len(sample_cols)==98 and set(sample_cols)==set(meta['sample'])
assert header[:4]==['gene_id','gene_name','gene_description','gene_locus']
raw = pd.read_csv(matrix_path,sep='\t',usecols=['gene_id','gene_name']+sample_cols,
                  compression='gzip',low_memory=False)
clean = raw['gene_name'].astype('string').str.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*').fillna(False)
values = raw.loc[clean,sample_cols].apply(pd.to_numeric,errors='coerce')
finite = np.isfinite(values.to_numpy()).all(axis=1)
nonnegative = (values.to_numpy()>=0).all(axis=1)
eligible = raw.loc[clean].loc[finite & nonnegative].copy()
eligible.loc[:,sample_cols] = values.loc[finite & nonnegative].to_numpy()
expr = eligible.groupby('gene_name',sort=True)[sample_cols].mean().astype(float)
assert expr.index.is_unique and np.isfinite(expr.to_numpy()).all() and (expr.to_numpy()>=0).all()
meta = meta.set_index('sample').loc[sample_cols].reset_index()

modules = pd.read_csv(ROOT/'modules/frozen_pca_modules.csv')
parameters = json.loads((ROOT/'metadata/stage22/primary_classifier_parameters.json').read_text())
frozen = modules.gene.tolist()
shared = modules.gene[modules.gene.isin(expr.index)].tolist()
assert len(shared)>4000

# Full-target z-scoring is label-free and transductive. No discovery probability
# calibration is claimed for the RNA-seq measurement scale.
logexpr = np.log2(expr.loc[shared]+1)
gene_sd = logexpr.std(axis=1,ddof=1).replace(0,np.nan)
z = pd.DataFrame(0.0,index=frozen,columns=sample_cols)
z.loc[shared] = logexpr.sub(logexpr.mean(axis=1),axis=0).div(gene_sd,axis=0).fillna(0)
score = pd.DataFrame(index=sample_cols)
for module,group in modules.groupby('module'):
    score[module] = (np.sign(group.loading_signed.to_numpy())
                     @ z.loc[group.gene].to_numpy()) / len(group)
score = score[parameters['features']]
x = (score.to_numpy()-np.array(parameters['scaler_mean']))/np.array(parameters['scaler_scale'])
linear = x@np.array(parameters['coefficients'][0])+parameters['intercept'][0]
prob = expit(linear)
pred = meta.copy()
pred['linear_score'] = linear
pred['prob_AD_uncalibrated'] = prob
for name in score.columns:
    pred[name] = score[name].to_numpy()
pred.to_csv(OUT/'GSE248417_frozen_predictions.csv',index=False)

y = pred.status.eq('AD').to_numpy(dtype=int)
auc = roc_auc_score(y,linear)
ap = average_precision_score(y,linear)
rng=np.random.default_rng(20261009)
case=np.flatnonzero(y==1); ctl=np.flatnonzero(y==0)
boots=[]
for _ in range(2000):
    ix=np.r_[rng.choice(case,len(case),replace=True),rng.choice(ctl,len(ctl),replace=True)]
    boots.append(roc_auc_score(y[ix],linear[ix]))

def hedges(a,b,axis=0):
    n1,n0=len(a),len(b)
    pool=np.sqrt(((n1-1)*a.var(axis=axis,ddof=1)+(n0-1)*b.var(axis=axis,ddof=1))/(n1+n0-2))
    return (1-3/(4*(n1+n0)-9))*(a.mean(axis=axis)-b.mean(axis=axis))/pool

effects={}
for name in ['M01','M07']:
    a=pred.loc[pred.status.eq('AD'),name].to_numpy()
    b=pred.loc[pred.status.eq('Control'),name].to_numpy()
    erng=np.random.default_rng(20261010)
    bs=[hedges(erng.choice(a,len(a),replace=True),erng.choice(b,len(b),replace=True)) for _ in range(2000)]
    effects[name]={'g':float(hedges(a,b)),'ci':np.quantile(bs,[.025,.975]).tolist()}

disc=pd.read_pickle(ROOT/'processed/GSE63060_gene_expression.pkl',compression='gzip')
dm=pd.read_csv(ROOT/'processed/GSE63060_matrix_formal_labels.csv').set_index('matrix_sample')
ad_disc=dm.index[dm.label_formal.eq('AD')]
ctl_disc=dm.index[dm.label_formal.eq('Control')]
ad_target=pred.loc[pred.status.eq('AD'),'sample']
ctl_target=pred.loc[pred.status.eq('Control'),'sample']
discg=hedges(disc.loc[shared,ad_disc].to_numpy().T,
             disc.loc[shared,ctl_disc].to_numpy().T)
targetg=hedges(logexpr.loc[shared,ad_target].to_numpy().T,
               logexpr.loc[shared,ctl_target].to_numpy().T)
gene=pd.DataFrame({'gene':shared,'discovery_g':discg,'GSE248417_g':targetg})
gene.to_csv(OUT/'GSE248417_gene_effects.csv',index=False)
good=gene.replace([np.inf,-np.inf],np.nan).dropna()
concordance={'n':len(good),
             'spearman_rho':float(spearmanr(good.discovery_g,good.GSE248417_g).statistic),
             'same_direction_fraction':float((np.sign(good.discovery_g)==np.sign(good.GSE248417_g)).mean())}

summary={'source':'GSE248417 mRNA gene-level DE supplementary file',
         'article_doi':'10.1186/s13195-026-01977-x',
         'matrix_sha256':sha(matrix_path),'series_sha256':sha(series_path),
         'source_rows':len(raw),'eligible_rows':len(eligible),'unique_symbols':len(expr),
         'samples':98,'AD':int(y.sum()),'controls':int((1-y).sum()),
         'frozen_genes_covered':len(shared),'frozen_gene_coverage_fraction':len(shared)/len(frozen),
         'auc_decision_score':auc,'auc_bootstrap_ci':np.quantile(boots,[.025,.975]).tolist(),
         'average_precision':ap,'auc_sigmoid_probability':roc_auc_score(y,prob),
         'module_effects':effects,'gene_effect_concordance':concordance,
         'method':'deposited per-sample nonnegative mRNA values; duplicated gene symbols averaged; log2(value+1); label-free full-target gene z-scores; original signed modules and classifier coefficients; decision-score ranking; no target refit or probability calibration',
         'source_count_note':'Article describes 50 AD/50 control; public GSE248417 mRNA file and series matrix link 49/49. Only those 98 are scored.'}
(OUT/'GSE248417_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
meta.to_csv(OUT/'GSE248417_linked_metadata.csv',index=False)
print(json.dumps(summary,indent=2))
