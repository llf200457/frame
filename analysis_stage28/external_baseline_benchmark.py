"""Exploratory external transfer of prespecified simple blood-gene baselines.

The candidate pool is restricted to genes with complete label-free availability
in all three series; panel sampling, variance ranking and classifier fits use
GSE63060 only. The exact same
trained model is scored in related GSE63061 and independent GSE140829.
Random panels are reported as a complete distribution, never selected by
external AUC. This is a transparent comparator, not an identical-capacity
replacement for the eight-module model.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis_stage28'
OUT.mkdir(exist_ok=True)
SEED = 20261008

disc = pd.read_pickle(ROOT/'processed/GSE63060_gene_expression.pkl', compression='gzip')
related = pd.read_pickle(ROOT/'processed/GSE63061_gene_expression.pkl', compression='gzip')
independent = pd.read_csv(ROOT/'raw/GSE140829/GSE140829_final_normalized_data.txt.gz',
                          sep='\t', index_col=0, compression='gzip')
pred = pd.read_csv(ROOT/'analysis_stage26/external_transport_predictions.csv')
cohorts = {}
for name, expression in [('GSE63060',disc), ('GSE63061',related), ('GSE140829',independent)]:
    endpoint = pred[(pred.cohort == name) & pred.status.isin(['AD','CTL'])].copy()
    assert endpoint['sample'].is_unique and set(endpoint['sample']) <= set(expression.columns)
    cohorts[name] = (expression, endpoint['sample'].tolist(), endpoint.status.eq('AD').to_numpy(dtype=int))
assert [(name,len(cohorts[name][1]),int(cohorts[name][2].sum())) for name in cohorts] == [
    ('GSE63060',249,145),('GSE63061',273,139),('GSE140829',427,198)]

common = disc.index.intersection(related.index).intersection(independent.index)
for name, (expression, ids, _) in cohorts.items():
    complete = np.isfinite(expression.loc[common,ids].to_numpy(dtype=float)).all(axis=1)
    common = common[complete]
assert len(common) > 10000
pool = common.to_numpy()
matrix = {name: expression.loc[common,ids].to_numpy(dtype=np.float32).T
          for name,(expression,ids,_) in cohorts.items()}
y = {name: cohorts[name][2] for name in cohorts}
assert all(np.isfinite(x).all() for x in matrix.values())

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(matrix['GSE63060'],y['GSE63060']))
rng = np.random.default_rng(SEED)
variance = disc.loc[common].var(axis=1,ddof=1).to_numpy()
top75 = np.argsort(variance)[-75:]
panels = [('variance_top75', top75)] + [
    (f'random_{i+1:03d}',rng.choice(len(common),size=75,replace=False)) for i in range(200)]
rows=[]
for name, columns in panels:
    xd = matrix['GSE63060'][:,columns]
    oof = np.empty(len(xd),float)
    for fit,test in folds:
        scaler=StandardScaler().fit(xd[fit])
        model=LogisticRegression(C=1,max_iter=2000,random_state=SEED)
        model.fit(scaler.transform(xd[fit]),y['GSE63060'][fit])
        # Decision scores preserve pairwise ranking when sigmoid probabilities
        # saturate numerically under cross-assay expression shifts.
        oof[test]=model.decision_function(scaler.transform(xd[test]))
    scaler=StandardScaler().fit(xd)
    model=LogisticRegression(C=1,max_iter=2000,random_state=SEED)
    model.fit(scaler.transform(xd),y['GSE63060'])
    result={'panel':name,'genes':'|'.join(pool[columns]),
            'GSE63060_oof_auc':roc_auc_score(y['GSE63060'],oof),
            'GSE63060_oof_ap':average_precision_score(y['GSE63060'],oof)}
    for target in ['GSE63061','GSE140829']:
        score=model.decision_function(scaler.transform(matrix[target][:,columns]))
        result[target+'_auc']=roc_auc_score(y[target],score)
        result[target+'_ap']=average_precision_score(y[target],score)
    result['independent_minus_related_auc']=result['GSE140829_auc']-result['GSE63061_auc']
    rows.append(result)
frame=pd.DataFrame(rows)
frame.to_csv(OUT/'external_baseline_panels.csv',index=False)
random=frame[frame.panel.str.startswith('random_')]
def q(values):
    return {str(k):float(v) for k,v in values.quantile([.05,.5,.95]).items()}
summary={'seed':SEED,'common_complete_gene_pool':len(common),'random_panel_n':200,
         'genes_per_panel':75,'cohort_n':{k:len(v[1]) for k,v in cohorts.items()},
         'random_quantiles':{col:q(random[col]) for col in ['GSE63060_oof_auc','GSE63061_auc',
                         'GSE140829_auc','independent_minus_related_auc']},
         'variance_top75':frame.iloc[0].drop('genes').to_dict(),
         'fraction_random_independent_auc_at_least_0_6':float((random.GSE140829_auc>=.6).mean()),
         'fraction_random_related_auc_at_least_0_7':float((random.GSE63061_auc>=.7).mean()),
         'design_note':'The candidate pool uses label-free three-cohort feature availability; panel sampling, variance ranking and classifier fits use discovery only. No external outcome selection. Expression scales are untouched GEO processed scales with discovery-fitted standardization. AUC/AP use decision_function scores to avoid sigmoid saturation under cross-assay shift. Random-panel quantiles describe candidate variability, not sample uncertainty.'}
(OUT/'external_baseline_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
