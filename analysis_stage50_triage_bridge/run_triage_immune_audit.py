"""New mechanistic/error audit of the ORIGINAL frozen blood triage framework.

Relative marker scores do not identify true immune-cell fractions. The marker
overlap mask is feature sensitivity, not a biological knockout. No external
diagnosis is used to fit feature definitions, classifier coefficients or gates.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import fisher_exact, spearmanr, t
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
SEED = 20261002
rng = np.random.default_rng(SEED)
MEMBERS = pd.read_csv(DATA/'modules/frozen_pca_modules.csv')
PARAMFILE = DATA/'metadata/stage22/primary_classifier_parameters.json'
PARAM = json.loads(PARAMFILE.read_text(encoding='utf-8'))
MARKERFILE = DATA/'analysis_stage22/resources/genes.txt'
MARKERS = pd.read_csv(MARKERFILE, sep='\t')
FEATURES = PARAM['features']
GENES = MEMBERS.gene.tolist()
CELL_TYPES = [c for c in MARKERS['Cell population'].unique() if c not in ['Endothelial cells', 'Fibroblasts']]

def adjustment(p, method='bh'):
    p = np.asarray(p, float)
    order = np.argsort(p)
    if method == 'bh':
        q = np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    else:
        q = np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    result = np.empty(len(p))
    result[order] = np.minimum(q, 1)
    return result

disc = pd.read_pickle(DATA/'processed/GSE63060_gene_expression.pkl', compression='gzip')
ext = pd.read_pickle(DATA/'processed/GSE63061_gene_expression.pkl', compression='gzip')
ind = pd.read_csv(DATA/'raw/GSE140829/GSE140829_final_normalized_data.txt.gz', sep='\t', index_col=0)
matrices = {'GSE63060':disc, 'GSE63061':ext, 'GSE140829':ind}
labels0 = pd.read_csv(DATA/'processed/GSE63060_matrix_formal_labels.csv').set_index('matrix_sample')
labels1 = pd.read_csv(DATA/'metadata/stage20/GSE63061_relabelled_predictions_all_samples.csv').set_index('matrix_sample')
labels2 = pd.read_csv(DATA/'analysis_stage26/GSE140829_linked_metadata.csv').set_index('expression_id')
annotations = {'GSE63060':labels0, 'GSE63061':labels1, 'GSE140829':labels2}
frozen_predictions = pd.read_csv(DATA/'analysis_stage26/external_transport_predictions.csv')
mu, sd = disc.loc[GENES].mean(axis=1), disc.loc[GENES].std(axis=1, ddof=1)

# Age/sex for the corrected GSE63061 sample mapping, not the old initial labels.
series = (DATA/'metadata/GSE63061_series_metadata.txt').read_text(encoding='utf-8')
import csv
fields = [next(csv.reader([line], delimiter='\t')) for line in series.splitlines() if line.startswith('!Sample_')]
geo = next(v[1:] for v in fields if v[0]=='!Sample_geo_accession')
for name in ['age', 'gender']:
    row = next(v[1:] for v in fields if v[0]=='!Sample_characteristics_ch1' and any(s.startswith(name+':') for s in v[1:]))
    mapping = dict(zip(geo, [v.split(':',1)[1].strip() for v in row]))
    labels1[name] = labels1.geo_accession.map(mapping)

common = set.intersection(*(set(x.index) for x in matrices.values()))
signatures = {cell:sorted(set(MARKERS.loc[MARKERS['Cell population']==cell,'HUGO symbols'])&common) for cell in CELL_TYPES}
mask = set().union(*signatures.values())
coverage = []
for cell, genes in signatures.items():
    coverage.append({'cell':cell, 'signature_n':int(MARKERS.loc[MARKERS['Cell population']==cell,'HUGO symbols'].nunique()),
                     'common_n':len(genes), 'genes':';'.join(genes), 'frozen_module_overlap_n':len(set(genes)&set(GENES)),
                     'score_eligible':len(genes)>=2})
pd.DataFrame(coverage).to_csv(ROOT/'immune_marker_coverage.csv', index=False)
signatures = {cell:genes for cell,genes in signatures.items() if len(genes)>=2}
CELL_TYPES = list(signatures)
GATE_TYPES = ['Monocytic lineage', 'Neutrophils']
assert all(cell in signatures for cell in GATE_TYPES)

def module_scores(matrix, remove_marker_overlap=False):
    shared = [g for g in GENES if g in matrix.index and (not remove_marker_overlap or g not in mask)]
    z = pd.DataFrame(0., index=GENES, columns=matrix.columns)
    z.loc[shared] = matrix.loc[shared].sub(mu.loc[shared], axis=0).div(sd.loc[shared], axis=0).fillna(0)
    return pd.DataFrame({m:np.sign(part.loading_signed.to_numpy())@z.loc[part.gene].to_numpy()/len(part)
                         for m,part in MEMBERS.groupby('module')}, index=matrix.columns)[FEATURES]

def probability(modules):
    scaled = (modules.to_numpy()-np.array(PARAM['scaler_mean']))/np.array(PARAM['scaler_scale'])
    return expit(scaled@np.array(PARAM['coefficients'][0])+PARAM['intercept'][0])

def covariates(frame, cohort):
    if cohort=='GSE63060': age, sex = frame['age:ch1'], frame['gender:ch1']
    elif cohort=='GSE63061': age, sex = frame['age'], frame['gender']
    else: age, sex = frame['age_at_draw'], frame['sex']
    age = pd.to_numeric(age, errors='coerce').to_numpy(float)
    sex = sex.astype(str).str.lower().str.startswith('m').to_numpy(float)
    return age, sex

def partial_rank(a, b, y, age, sex):
    valid = np.isfinite(a)&np.isfinite(b)&np.isfinite(age)&np.isfinite(sex)
    a,b,y,age,sex = a[valid],b[valid],y[valid],age[valid],sex[valid]
    ra,rb = pd.Series(a).rank().to_numpy(),pd.Series(b).rank().to_numpy()
    c = np.column_stack([np.ones(len(y)),y,pd.Series(age).rank().to_numpy(),sex])
    rank = np.linalg.matrix_rank(c)
    ar = ra-c@np.linalg.lstsq(c,ra,rcond=None)[0]
    br = rb-c@np.linalg.lstsq(c,rb,rcond=None)[0]
    rho = float(np.corrcoef(ar,br)[0,1])
    df = len(y)-rank-1
    p = float(2*t.sf(abs(rho)*np.sqrt(df/max(1e-15,1-rho*rho)),df))
    return {'partial_rank_r':rho,'p':p,'n_complete':len(y),'covariate_rank':int(rank)}

immune = {cohort:pd.DataFrame({cell:x.loc[genes].mean(axis=0) for cell,genes in signatures.items()}) for cohort,x in matrices.items()}
thresholds = {cell:immune['GSE63060'][cell].quantile([.05,.95]).tolist() for cell in CELL_TYPES}
disc_modules = module_scores(disc)
dcov = np.cov(disc_modules.to_numpy(), rowvar=False)
dinv = np.linalg.pinv(dcov)
dm = disc_modules.mean(axis=0).to_numpy()
def distance(m):
    a = m.to_numpy()-dm
    return np.einsum('ij,jk,ik->i',a,dinv,a)
drift_cutoff = float(np.quantile(distance(disc_modules),.95))
associations, metrics, enrichment, predictions = [],[],[],[]
for cohort, matrix in matrices.items():
    print('Auditing original triage:',cohort, flush=True)
    complete = module_scores(matrix)
    masked = module_scores(matrix, remove_marker_overlap=True)
    annotation = annotations[cohort].loc[matrix.columns]
    if cohort=='GSE63060': status = annotation.label_formal.replace({'Control':'CTL'})
    elif cohort=='GSE63061': status = annotation.status
    else: status = annotation.diagnosis.replace({'Control':'CTL'})
    keep = status.isin(['AD','CTL'])
    ids = matrix.columns[keep]
    y = status.loc[ids].eq('AD').to_numpy(int)
    m, mm = complete.loc[ids],masked.loc[ids]
    p,pm = probability(m),probability(mm)
    previous = frozen_predictions.loc[(frozen_predictions.cohort==cohort)&frozen_predictions['sample'].isin(ids)].set_index('sample')
    assert len(previous)==len(ids)
    assert np.max(np.abs(previous.loc[ids,'prob_AD'].to_numpy()-p))<1e-10
    age,sex = covariates(annotation.loc[ids],cohort)
    for module in ['M01','M07']:
        for cell in ['Monocytic lineage','Neutrophils']:
            a,b = m[module].to_numpy(),immune[cohort].loc[ids,cell].to_numpy()
            am = mm[module].to_numpy()
            rawr,rawp = spearmanr(a,b)
            maskedr,_ = spearmanr(am,b)
            associations.append({'cohort':cohort,'module':module,'cell':cell,'n':len(y),'raw_rho':rawr,'raw_p':rawp,
                                 'marker_masked_rho':maskedr,'masked_module_genes':int(MEMBERS.loc[MEMBERS.module==module,'gene'].isin(mask).sum()),
                                 **partial_rank(am,b,y,age,sex)})
    score = np.log(np.clip(p,1e-12,1-1e-12)/(1-np.clip(p,1e-12,1-1e-12)))
    errors = (p>=.5)!=y
    sqerr = (p-y)**2
    dist = distance(m)
    drift = dist>drift_cutoff
    gate = np.column_stack([(immune[cohort].loc[ids,c].to_numpy()<thresholds[c][0])|(immune[cohort].loc[ids,c].to_numpy()>thresholds[c][1]) for c in GATE_TYPES]).any(axis=1)
    frame = pd.DataFrame({'sample':ids,'cohort':cohort,'status':status.loc[ids].to_numpy(),'p_frozen':p,'p_marker_masked':pm,
                          'hard_error':errors,'squared_error':sqerr,'module_drift':dist,'drift_alert':drift,
                          'immune_extreme':gate,'candidate_defer_with_immune':drift|gate})
    for c in ['Monocytic lineage','Neutrophils']: frame[c]=immune[cohort].loc[ids,c].to_numpy()
    predictions.append(frame)
    immune[cohort].to_csv(ROOT/f'{cohort}_immune_relative_scores.csv', encoding='utf-8-sig')
    for mapping, pred in [('original_frozen',p),('immune_marker_overlap_mask',pm)]:
        metrics.append({'cohort':cohort,'mapping':mapping,'n':len(y),'AD':int(y.sum()),'auc':roc_auc_score(y,pred),
                        'ap':average_precision_score(y,pred),'brier':brier_score_loss(y,pred),
                        'mean_probability':float(pred.mean()),'mean_logit':float(np.log(pred/(1-pred)).mean())})
    for region in ['all','immune_extreme','immune_reference_range','drift_accepted','drift_rejected']:
        flag = np.ones(len(y),bool) if region=='all' else gate if region=='immune_extreme' else ~gate if region=='immune_reference_range' else ~drift if region=='drift_accepted' else drift
        yy,pp = y[flag],p[flag]
        metrics.append({'cohort':cohort,'mapping':'original_frozen/'+region,'n':len(yy),'AD':int(yy.sum()),
                        'auc':roc_auc_score(yy,pp) if len(set(yy))==2 else None,
                        'brier':brier_score_loss(yy,pp) if len(yy) else None,
                        'hard_error_rate':float(errors[flag].mean()) if flag.any() else None,
                        'acceptance_rate':float(flag.mean())})
    if cohort!='GSE63060':
        table = np.array([[np.sum(errors&gate),np.sum(~errors&gate)],[np.sum(errors&~gate),np.sum(~errors&~gate)]])
        odds,pvalue = fisher_exact(table)
        diffs=[]
        for _ in range(1000):
            idx=rng.choice(len(y),len(y),replace=True)
            gg=gate[idx]
            if gg.any() and (~gg).any(): diffs.append(sqerr[idx][gg].mean()-sqerr[idx][~gg].mean())
        enrichment.append({'cohort':cohort,'extreme_n':int(gate.sum()),'reference_n':int((~gate).sum()),
                           'error_odds_ratio':float(odds) if np.isfinite(odds) else None,'fisher_p':pvalue,
                           'brier_difference_extreme_minus_reference':float(sqerr[gate].mean()-sqerr[~gate].mean()) if gate.any() and (~gate).any() else None,
                           'brier_difference_ci':np.quantile(diffs,[.025,.975]).tolist() if diffs else None,
                           'all_drift_rejected':bool(drift.all()),
                           'fisher_test_estimable':bool(gate.any() and (~gate).any())})

associations = pd.DataFrame(associations)
associations['q_bh'] = adjustment(associations.p)
associations.to_csv(ROOT/'module_immune_overlap_control_associations.csv', index=False)
pd.DataFrame(metrics).to_csv(ROOT/'triage_immune_sensitivity_metrics.csv', index=False)
enrichment=pd.DataFrame(enrichment)
enrichment['p_holm']=adjustment(enrichment.fisher_p,'holm')
enrichment.to_csv(ROOT/'triage_error_immune_enrichment.csv', index=False)
pd.concat(predictions).to_csv(ROOT/'triage_immune_audit_predictions.csv', index=False)

protocol=json.loads((ROOT/'execution_protocol.json').read_text(encoding='utf-8'))
candidates=protocol['genetic_extension']['candidate_genes']
gene_registry=[]
marker_gene_map=MARKERS.groupby('HUGO symbols')['Cell population'].agg(lambda s:';'.join(sorted(set(s)))).to_dict()
for gene in candidates:
    row=MEMBERS.loc[MEMBERS.gene==gene]
    gene_registry.append({'gene':gene,'in_frozen_5000':len(row)==1,'module':row.module.iloc[0] if len(row) else None,
                          'frozen_loading_sign':float(np.sign(row.loading_signed.iloc[0])) if len(row) else None,
                          'immune_marker_groups':marker_gene_map.get(gene,''),
                          'selection_basis':'existing M01/M07 interpretation or prespecified NF-kappaB regulatory context; not GWAS outcome',
                          'valid_genetic_instrument_confirmed':False})
pd.DataFrame(gene_registry).to_csv(ROOT/'prespecified_genetic_candidate_module_links.csv',index=False)

source_files=[PARAMFILE,MARKERFILE,DATA/'modules/frozen_pca_modules.csv',ROOT/'execution_protocol.json',Path(__file__)]
manifest={'date':'2026-10-02','seed':SEED,'bootstrap_n':1000,'classifier_refitted':False,'external_labels_used_for_feature_or_gate_fitting':False,
          'probabilities_reconstructed_within_1e_10':True,'cell_score_type':'relative marker abundance, not cell fractions',
          'signature_source':'https://github.com/ebecht/MCPcounter','common_marker_union_n':len(mask),
          'module_immune_marker_overlap_n':len(mask&set(GENES)),'drift_threshold':drift_cutoff,
          'immune_thresholds_from_discovery':thresholds,'immune_extreme_gate_types':GATE_TYPES,'association_family_n':len(associations),
          'markers_insufficient_groups_excluded_before_outcome_analysis':True,
          'frozen_classifier_and_initial_triage_labels_are_not_mixed':True,
          'sources_sha256':{str(p.relative_to(DATA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
          'interpretation':'Post-inspection explanatory audit; no new clinical validity, causal effect or biological knockout established.'}
(ROOT/'triage_immune_run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(associations[['cohort','module','cell','partial_rank_r','q_bh']].to_string(index=False))
print(enrichment.to_string(index=False))
