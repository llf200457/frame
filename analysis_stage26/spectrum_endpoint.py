"""Independent-cohort endpoint broadening: AD versus CTL plus MCI."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE if (BASE/'modules').is_dir() else BASE.parents[1]
OUT=ROOT/'analysis_stage26'
a=pd.read_csv(OUT/'external_transport_predictions.csv')
a=a[(a.cohort=='GSE140829')&(a.status.isin(['AD','CTL','MCI']))]
y=a.status.eq('AD').to_numpy().astype(int)
p=a.prob_AD.to_numpy()
rng=np.random.default_rng(20260929)
case=np.flatnonzero(y==1);control=np.flatnonzero(y==0)
draws=[]
for _ in range(2000):
    idx=np.r_[rng.choice(case,len(case),replace=True),rng.choice(control,len(control),replace=True)]
    draws.append(roc_auc_score(y[idx],p[idx]))
out={'endpoint':'GSE140829 AD versus CTL + MCI','n':len(a),'AD_n':int(y.sum()),
     'CTL_n':int(a.status.eq('CTL').sum()),'MCI_n':int(a.status.eq('MCI').sum()),
     'auc':roc_auc_score(y,p),'auc_bootstrap_ci':np.quantile(draws,[.025,.975]).tolist(),
     'ap':average_precision_score(y,p),'brier':brier_score_loss(y,p),
     'note':'All probabilities from the original frozen model; MCI is non-AD comparator only for this secondary endpoint.'}
(OUT/'spectrum_endpoint.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
