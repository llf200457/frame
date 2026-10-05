"""Exploratory target-cohort context: within-batch ranking and covariate increment.

Models with target labels are fitted only here, in five-fold out-of-fold
evaluation. They are NOT frozen external validations or clinical deployment.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,brier_score_loss,average_precision_score

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE if (BASE/'modules').is_dir() else BASE.parents[1]
OUT=ROOT/'analysis_stage26'
meta=pd.read_csv(OUT/'GSE140829_linked_metadata.csv').set_index('expression_id')
pred=pd.read_csv(OUT/'external_transport_predictions.csv')
pred=pred[(pred.cohort=='GSE140829')&pred.status.isin(['AD','CTL'])].set_index('sample')
frame=pred.join(meta[['age_at_draw','sex','apoe_status','batch']])
assert len(frame)==427 and frame[['age_at_draw','sex','batch']].notna().all().all()
frame['apoe4_missing']=frame.apoe_status.isna().astype(int)
frame['apoe4_carrier']=frame.apoe_status.fillna('').str.contains('E4').astype(int)
frame['male']=frame.sex.eq('Male').astype(int)
frame['frozen_logit']=logit(frame.prob_AD.clip(1e-6,1-1e-6))
y=frame.status.eq('AD').to_numpy().astype(int)
base=['age_at_draw','male','apoe4_carrier','apoe4_missing']
folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260929)
out={k:np.empty(len(frame)) for k in ['clinical','clinical_plus_frozen']}
for fit,test in folds.split(frame,y):
    for name,cols in [('clinical',base),('clinical_plus_frozen',base+['frozen_logit'])]:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=20260929))
        model.fit(frame.iloc[fit][cols],y[fit])
        out[name][test]=model.predict_proba(frame.iloc[test][cols])[:,1]
for name,values in out.items():
    frame[name+'_oof']=values
frame.to_csv(OUT/'GSE140829_clinical_increment_oof.csv')

def auc_within_batch(tab, rng=None):
    total=0.;pairs=0
    for _,group in tab.groupby('batch'):
        a=group.loc[group.status.eq('AD'),'prob_AD'].to_numpy()
        b=group.loc[group.status.eq('CTL'),'prob_AD'].to_numpy()
        if not len(a) or not len(b):continue
        if rng is not None:
            a=rng.choice(a,len(a),replace=True)
            b=rng.choice(b,len(b),replace=True)
        total+=(a[:,None]>b).sum()+.5*(a[:,None]==b).sum()
        pairs+=len(a)*len(b)
    return total/pairs,pairs

rng=np.random.default_rng(20260929)
case=np.flatnonzero(y==1);control=np.flatnonzero(y==0)
draws=[]
for _ in range(2000):
    ix=np.r_[rng.choice(case,len(case),replace=True),rng.choice(control,len(control),replace=True)]
    yi=y[ix];c=out['clinical'][ix];t=out['clinical_plus_frozen'][ix]
    draws.append([roc_auc_score(yi,t)-roc_auc_score(yi,c),
                  brier_score_loss(yi,t)-brier_score_loss(yi,c)])
draws=np.asarray(draws)
within,pairs=auc_within_batch(frame)
within_rng=np.random.default_rng(20260930)
within_draws=[auc_within_batch(frame,within_rng)[0] for _ in range(2000)]
summary={
  'endpoint':'GSE140829 AD versus Control; exploratory target-cohort five-fold out-of-fold fitting',
  'n':len(frame),'AD_n':int(y.sum()),'CTL_n':int((1-y).sum()),
  'covariates':base,'missing_APOE_n':int(frame.apoe4_missing.sum()),
  'clinical_oof':{'auc':roc_auc_score(y,out['clinical']),'ap':average_precision_score(y,out['clinical']),
                  'brier':brier_score_loss(y,out['clinical'])},
  'clinical_plus_frozen_oof':{'auc':roc_auc_score(y,out['clinical_plus_frozen']),
                  'ap':average_precision_score(y,out['clinical_plus_frozen']),
                  'brier':brier_score_loss(y,out['clinical_plus_frozen'])},
  'delta_auc_plus_minus_clinical':float(roc_auc_score(y,out['clinical_plus_frozen'])-roc_auc_score(y,out['clinical'])),
  'delta_auc_bootstrap_ci':np.quantile(draws[:,0],[.025,.975]).tolist(),
  'delta_brier_plus_minus_clinical':float(brier_score_loss(y,out['clinical_plus_frozen'])-brier_score_loss(y,out['clinical'])),
  'delta_brier_bootstrap_ci':np.quantile(draws[:,1],[.025,.975]).tolist(),
  'within_batch_auc_frozen':within,'within_batch_auc_bootstrap_ci':np.quantile(within_draws,[.025,.975]).tolist(),
  'within_batch_case_control_pairs':pairs,
  'overall_auc_frozen':roc_auc_score(y,frame.prob_AD),
  'design_note':'Covariate models are estimated within GSE140829 and assessed by target-cohort OOF predictions; not an external validation of the combined model.'}
(OUT/'clinical_increment_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
