"""Post-review external transport of the archived fixed eight-module classifier.

No labels from either new cohort enter feature mapping, scaling, or scoring.
Missing frozen genes are set to the discovery mean on its standardized scale
(zero), with the ORIGINAL module denominator retained. This is a documented
assay-availability adaptation, not a retrained classifier.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE if (BASE/'modules').is_dir() else BASE.parents[1]
OUT = ROOT / 'analysis_stage26'
OUT.mkdir(exist_ok=True)
SEED = 20260929
M = pd.read_csv(ROOT/'modules/frozen_pca_modules.csv')
PARAM = json.loads((ROOT/'overleaf_upload_route2_v7/source_data/stage22/primary_classifier_parameters.json').read_text())
COLS = PARAM['features']
DISC = pd.read_pickle(ROOT/'processed/GSE63060_gene_expression.pkl', compression='gzip')
OLD = pd.read_pickle(ROOT/'processed/GSE63061_gene_expression.pkl', compression='gzip')
FROZEN = M.gene.tolist()
assert len(FROZEN) == len(set(FROZEN)) == 5000
assert set(FROZEN) <= set(DISC.index) and set(FROZEN) <= set(OLD.index)
MU = DISC.loc[FROZEN].mean(axis=1)
SD = DISC.loc[FROZEN].std(axis=1, ddof=1)
assert (SD > 0).all()

def scores(x, target_relative=False):
    genes = list(set(FROZEN)&set(x.index))
    z = pd.DataFrame(0., index=FROZEN, columns=x.columns)
    if target_relative:
        m = x.loc[genes].mean(axis=1)
        s = x.loc[genes].std(axis=1,ddof=1).replace(0,np.nan)
        z.loc[genes] = x.loc[genes].sub(m,axis=0).div(s,axis=0).fillna(0)
    else:
        z.loc[genes] = x.loc[genes].sub(MU.loc[genes],axis=0).div(SD.loc[genes],axis=0).fillna(0)
    result = pd.DataFrame(index=x.columns)
    for k, group in M.groupby('module'):
        w = np.sign(group.loading_signed.to_numpy()) / len(group)
        result[k] = w @ z.loc[group.gene].to_numpy()
    return result[COLS]

def predict(z):
    x=(z.to_numpy()-np.array(PARAM['scaler_mean']))/np.array(PARAM['scaler_scale'])
    return expit(x @ np.array(PARAM['coefficients'][0])+PARAM['intercept'][0])

def metrics(y,p,seed=SEED,boots=2000):
    y=np.asarray(y,int);p=np.asarray(p,float)
    rng=np.random.default_rng(seed)
    cases=np.flatnonzero(y==1);controls=np.flatnonzero(y==0)
    assert len(cases)>0 and len(controls)>0
    original=np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)])
    draws=np.empty((boots,3))
    for i in range(boots):
        idx=np.r_[rng.choice(cases,len(cases),replace=True),rng.choice(controls,len(controls),replace=True)]
        draws[i]=[roc_auc_score(y[idx],p[idx]),average_precision_score(y[idx],p[idx]),brier_score_loss(y[idx],p[idx])]
    low,high=np.quantile(draws,[.025,.975],axis=0)
    return dict(zip(('auc','ap','brier'),original))|dict(zip(('auc_low','ap_low','brier_low'),low))|dict(zip(('auc_high','ap_high','brier_high'),high))

# The old cohort is a non-negotiable reconstruction check before new estimates.
reconstructed = scores(OLD)
stored = pd.read_csv(ROOT/'modules/GSE63061_frozen_module_scores.csv',index_col=0)
score_error=float((reconstructed-stored.loc[reconstructed.index,COLS]).abs().to_numpy().max())
archive = pd.read_csv(ROOT/'metadata/stage20/GSE63061_relabelled_predictions_all_samples.csv').set_index('matrix_sample')
prediction_error=float(np.max(np.abs(predict(reconstructed.loc[archive.index])-archive.prob_AD.to_numpy())))
assert score_error < 2e-12 and prediction_error < 1e-10, (score_error,prediction_error)

new1=pd.read_csv(ROOT/'raw/GSE140829/GSE140829_final_normalized_data.txt.gz',sep='\t',index_col=0,compression='gzip')
meta1=pd.read_csv(OUT/'GSE140829_linked_metadata.csv').set_index('expression_id')
assert set(new1.columns)<=set(meta1.index)
new2raw=pd.read_csv(ROOT/'raw/GSE97760/GSE97760_loess.txt.gz',sep='\t')
valuecols=[c for c in new2raw if c.startswith('Norm_')]
new2raw=new2raw.dropna(subset=['GeneSymbol'])
new2raw=new2raw[new2raw.GeneSymbol.str.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*')]
new2=np.log2(new2raw[valuecols].astype(float)+1)
new2.index=new2raw.GeneSymbol
new2=new2.groupby(level=0).mean()
meta2=pd.read_csv(OUT/'GSE97760_linked_metadata.csv').set_index('matrix_column')
assert set(new2.columns)==set(meta2.index)

datasets={
    'GSE63060':(DISC,pd.read_csv(ROOT/'processed/GSE63060_matrix_formal_labels.csv').set_index('matrix_sample')),
    'GSE63061':(OLD,archive),
    'GSE140829':(new1,meta1),
    'GSE97760':(new2,meta2),
}
rows=[];preds=[];cover=[];all_scores={}
for accession,(x,meta) in datasets.items():
    common=set(FROZEN)&set(x.index)
    cov={'cohort':accession,'mapped_genes':len(x),'frozen_genes_measured':len(common),
         'frozen_genes_missing':5000-len(common),'all_samples':len(x.columns)}
    for k,group in M.groupby('module'):
        cov[k+'_coverage']=len(set(group.gene)&common)/len(group)
    cover.append(cov)
    s=scores(x);all_scores[accession]=s
    p=predict(s)
    assert np.isfinite(p).all(), (accession, int((~np.isfinite(p)).sum()), s.isna().sum().to_dict())
    if accession=='GSE63060':
        lab=meta.loc[s.index,'label_formal'].replace({'Control':'CTL'})
    elif accession=='GSE63061':
        lab=meta.loc[s.index,'status']
    elif accession=='GSE140829':
        lab=meta.loc[s.index,'diagnosis'].replace({'Control':'CTL'})
    else:
        lab=meta.loc[s.index,'disease'].map({'healthy':'CTL',"advanced Alzheimer's disease":'AD'})
    record=pd.DataFrame({'cohort':accession,'sample':s.index,'status':lab.to_numpy(),'prob_AD':p})
    for k in COLS:record[k]=s[k].to_numpy()
    preds.append(record)
    eligible=record[record.status.isin(['AD','CTL'])].copy()
    y=eligible.status.eq('AD').to_numpy().astype(int)
    result=metrics(y,eligible.prob_AD)
    result.update({'cohort':accession,'method':'frozen-discovery-scale/zero-missing',
                   'n':len(eligible),'AD_n':int(y.sum()),'CTL_n':int((1-y).sum()),
                   'prob_min':float(eligible.prob_AD.min()),'prob_median':float(eligible.prob_AD.median()),
                   'prob_max':float(eligible.prob_AD.max())})
    rows.append(result)
    if accession in ['GSE140829','GSE97760']:
        alt=scores(x,target_relative=True)
        altp=predict(alt.loc[eligible['sample']])
        ar=metrics(y,altp)
        ar.update({'cohort':accession,'method':'target-relative-gene-standardization',
                   'n':len(eligible),'AD_n':int(y.sum()),'CTL_n':int((1-y).sum()),
                   'prob_min':float(altp.min()),'prob_median':float(np.median(altp)),
                   'prob_max':float(altp.max())})
        rows.append(ar)
        altout=eligible[['sample','status']].copy()
        altout['prob_AD']=altp
        altout.to_csv(OUT/f'{accession}_target_relative_predictions.csv',index=False)

# Isolate the effect of unavailable GSE140829 features in the old target batch.
mask=set(FROZEN)-set(new1.index)
old_masked=OLD.drop(index=list(mask))
masked=scores(old_masked)
oldlab=archive.loc[masked.index,'status']
keep=oldlab.isin(['AD','CTL'])
y=oldlab.loc[keep].eq('AD').to_numpy().astype(int)
masked_result=metrics(y,predict(masked.loc[keep]))
masked_result.update({'cohort':'GSE63061','method':'frozen-score/GSE140829-availability-mask',
                      'n':len(y),'AD_n':int(y.sum()),'CTL_n':int((1-y).sum()),
                      'prob_min':float(predict(masked.loc[keep]).min()),
                      'prob_median':float(np.median(predict(masked.loc[keep]))),
                      'prob_max':float(predict(masked.loc[keep]).max())})
rows.append(masked_result)
masked_predictions=pd.DataFrame({'sample':masked.index,'status':oldlab.to_numpy(),
                                 'prob_full':predict(reconstructed.loc[masked.index]),
                                 'prob_masked':predict(masked)})
masked_predictions.to_csv(OUT/'GSE63061_availability_mask_predictions.csv',index=False)
audited=masked_predictions[masked_predictions.status.isin(['AD','CTL'])]
case=audited[audited.status=='AD'];control=audited[audited.status=='CTL']
full_order=case.prob_full.to_numpy()[:,None]>control.prob_full.to_numpy()[None,:]
mask_order=case.prob_masked.to_numpy()[:,None]>control.prob_masked.to_numpy()[None,:]
pairwise_order_changes=int((full_order!=mask_order).sum())

# Independent-cohort performance gap: resample each dataset and class separately.
old_verified=archive[archive.status.isin(['AD','CTL'])]
new_verified=preds[2][preds[2].status.isin(['AD','CTL'])]
gap_rng=np.random.default_rng(20261001)
auc_gaps=[]
for _ in range(2000):
    draw=[]
    for tab in (old_verified,new_verified):
        c=tab[tab.status=='AD'];h=tab[tab.status=='CTL']
        sampled=pd.concat([c.iloc[gap_rng.integers(len(c),size=len(c))],
                           h.iloc[gap_rng.integers(len(h),size=len(h))]])
        draw.append(roc_auc_score(sampled.status.eq('AD'),sampled.prob_AD))
    auc_gaps.append(draw[1]-draw[0])
auc_gap=float(roc_auc_score(new_verified.status.eq('AD'),new_verified.prob_AD)-
              roc_auc_score(old_verified.status.eq('AD'),old_verified.prob_AD))
auc_gap_ci=np.quantile(auc_gaps,[.025,.975]).tolist()

pd.DataFrame(rows).to_csv(OUT/'external_transport_metrics.csv',index=False)
pd.concat(preds,ignore_index=True).to_csv(OUT/'external_transport_predictions.csv',index=False)
pd.DataFrame(cover).to_csv(OUT/'module_coverage.csv',index=False)
(OUT/'transport_audit.json').write_text(json.dumps({
    'frozen_module_reconstruction_max_error':score_error,
    'archived_probability_reconstruction_max_error':prediction_error,
    'GSE140829_missing_genes':len(mask),
    'GSE63061_case_control_pairwise_order_changes_after_mask':pairwise_order_changes,
    'GSE63061_case_control_pairs':len(case)*len(control),
    'GSE140829_minus_GSE63061_auc':auc_gap,
    'independent_two_cohort_auc_difference_bootstrap_ci':auc_gap_ci,
    'independent_two_cohort_auc_difference_bootstrap':'2000 independent class-stratified draws per cohort; seed 20261001; fitted predictions held fixed',
    'GSE97760_expression_processing':'log2(loess-normalized intensity+1); duplicate gene symbols averaged; no label-based normalization',
    'GSE97760_sparse_missing_values':'five frozen-gene/sample values across two controls were set to discovery-standardized zero',
    'missing_gene_policy':'zero on discovery-standardized gene scale; original denominator per module',
    'target_relative_policy':'secondary label-free transductive sensitivity; all target samples used to estimate gene mean and SD',
    'bootstrap':'2000 class-stratified draws; seed 20260929; fitted model held fixed',
    'no_retraining_on_external_labels':True,
},indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print(pd.DataFrame(cover).to_string(index=False,float_format=lambda x:f'{x:.3f}'))
