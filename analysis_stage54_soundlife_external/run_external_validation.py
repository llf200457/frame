"""One finite external experiment with fixed source predictor and calibration roles."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
threadpool_limits(limits=1)
OUT=Path(__file__).resolve().parent
DATA=OUT.parent
SOURCE=DATA/'analysis_stage53_measured_blood_anchor'
PROTOCOL=json.loads((OUT/'execution_protocol.json').read_text(encoding='utf-8'))
SEED=PROTOCOL['seed']


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for v in iter(lambda:f.read(1024*1024),b''):h.update(v)
    return h.hexdigest()


preserved={n:sha(DATA/n) for n in PROTOCOL['preserve_files']}
modelpath=SOURCE/'locked_neutrophil_factor_predictor.npz'
assert sha(modelpath)==PROTOCOL['expected_model_sha256']
model=np.load(modelpath,allow_pickle=False)
paired=pd.read_csv(OUT/'all_paired_visit_metadata.csv')
primary=pd.read_csv(OUT/'primary_one_visit_metadata.csv')
expr=pd.read_pickle(OUT/'SoundLife_model_gene_TPM.pkl',compression='gzip')
assert paired['sample'].is_unique and primary.subject.is_unique and set(expr.columns)==set(paired['sample'])
names=paired['sample'].to_list()
genes=model['genes'].tolist()
available=[g for g in genes if g in expr.index]
positions={g:i for i,g in enumerate(genes)}
z=np.zeros((len(names),len(genes)))
ix=np.array([positions[g] for g in available])
z[:,ix]=(np.log2(expr.loc[available,names].to_numpy(float).T+1)-model['gene_mean'][ix])/model['gene_sd'][ix]
modules=z@model['module_weights']
x=np.column_stack([paired[['age','male','sex_missing']].to_numpy(float),modules])
unclipped=((x-model['feature_mean'])/model['feature_scale'])@model['coefficients']+model['intercept']
locked=np.clip(unclipped,0.0,100.0)
source_meta=pd.read_csv(SOURCE/'paired_clinical_metadata.csv')
source_y=source_meta.target_neutrophils_percent.to_numpy(float)
source_cov=source_meta[['age','male','sex_missing']].to_numpy(float)
cov_scale=StandardScaler().fit(source_cov)
clinical_model=Ridge(alpha=10.0).fit(cov_scale.transform(source_cov),source_y)
clinical_pred=np.clip(clinical_model.predict(cov_scale.transform(paired[['age','male','sex_missing']].to_numpy(float))),0,100)
source_mean=float(source_y.mean())
base=pd.DataFrame({'sample':names,'subject':paired.subject,'role':paired.role,'primary_one_visit':paired.primary_one_visit,
                   'visit':paired['sample.visitName_RNA'],'days_since_first_visit':paired['sample.daysSinceFirstVisit_RNA'],
                   'age':paired.age,'male':paired.male,'observed_neutrophils_percent':paired.target_neutrophils_percent,
                   'source_mean':source_mean,'source_age_sex':clinical_pred,'locked8':locked,
                   'locked8_unclipped':unclipped})
first=base.set_index('sample').loc[primary['sample']].reset_index()
assert len(first)==len(primary)
cal=first[first.role.eq('calibration_subject')]
assert len(cal)==20 and cal.subject.is_unique
held=first[first.role.eq('heldout_subject')]
assert len(held)==74 and set(cal.subject).isdisjoint(set(held.subject))
delta=float((cal.observed_neutrophils_percent-cal.locked8).mean())
affine=LinearRegression().fit(cal[['locked8']].to_numpy(),cal.observed_neutrophils_percent.to_numpy())
cal_mean=float(cal.observed_neutrophils_percent.mean())
base['calibration_mean']=cal_mean
base['intercept_calibrated8']=np.clip(base.locked8+delta,0,100)
base['affine_calibrated8']=np.clip(affine.predict(base[['locked8']].to_numpy()),0,100)
first=base.set_index('sample').loc[primary['sample']].reset_index()
held=first[first.role.eq('heldout_subject')].reset_index(drop=True)
base.to_csv(OUT/'all_visit_predictions.csv',index=False)
first.to_csv(OUT/'primary_one_visit_predictions.csv',index=False)
held.to_csv(OUT/'heldout_subject_predictions.csv',index=False)
pd.DataFrame(modules,index=names,columns=model['feature_names'][3:]).to_csv(OUT/'external_module_scores.csv',index_label='sample')
parameters=dict(calibration_subject_n=20,heldout_subject_n=len(held),intercept_delta=delta,
                affine_intercept=float(affine.intercept_),affine_slope=float(affine.coef_[0]),
                target_calibration_mean=cal_mean,source_mean=source_mean,
                calibration_predictor_input='one locked source prediction only',target_gene_scaling=False,
                source_age_sex=dict(mean=cov_scale.mean_.tolist(),scale=cov_scale.scale_.tolist(),coef=clinical_model.coef_.tolist(),intercept=float(clinical_model.intercept_)))
(OUT/'calibration_parameters.json').write_text(json.dumps(parameters,indent=2),encoding='utf-8')


def metrics(y,p):
    denom=np.sum((y-y.mean())**2)
    return dict(R2=float(1-np.sum((y-p)**2)/denom) if denom>0 else float('nan'),
                MAE=float(np.abs(y-p).mean()),RMSE=float(np.sqrt(np.mean((y-p)**2))),
                Spearman_r=float(spearmanr(y,p).statistic) if np.ptp(p)>0 and np.ptp(y)>0 else float('nan'),
                bias=float((p-y).mean()))


rng=np.random.default_rng(SEED+54)
rows=[]
for scope,frame,models in [('external_primary_all94',first,['source_mean','source_age_sex','locked8']),
                           ('calibration_heldout74',held,['source_mean','source_age_sex','locked8','calibration_mean','intercept_calibrated8','affine_calibrated8'])]:
    draws=[rng.integers(0,len(frame),len(frame)) for _ in range(2000)]
    y=frame.observed_neutrophils_percent.to_numpy(float)
    for name in models:
        p=frame[name].to_numpy(float)
        row=dict(scope=scope,model=name,n=len(y),**metrics(y,p))
        boot=[metrics(y[i],p[i]) for i in draws]
        for key in ['R2','MAE','bias']:
            row[key+'_CI_low'],row[key+'_CI_high']=np.quantile([b[key] for b in boot],[.025,.975])
        rows.append(row)
        print(scope,name,metrics(y,p),flush=True)
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'external_prediction_performance.csv',index=False)
contrasts=[]
for scope,frame,baseline,extended in [('external_primary_all94',first,'source_age_sex','locked8'),
                                    ('calibration_heldout74',held,'calibration_mean','intercept_calibrated8'),
                                    ('calibration_heldout74',held,'calibration_mean','affine_calibrated8')]:
    y=frame.observed_neutrophils_percent.to_numpy(float)
    d=np.abs(y-frame[baseline].to_numpy(float))-np.abs(y-frame[extended].to_numpy(float))
    lo,hi=np.quantile([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)],[.025,.975])
    contrasts.append(dict(scope=scope,baseline=baseline,extended=extended,MAE_reduction=float(d.mean()),CI_low=float(lo),CI_high=float(hi)))
pd.DataFrame(contrasts).to_csv(OUT/'paired_MAE_increments.csv',index=False)

# Longitudinal diagnostics on 74 held-out subjects, with whole-donor resampling.
long=base[base.role.eq('heldout_subject')].copy().reset_index(drop=True)
groups=[v.to_numpy(int) for v in long.groupby('subject',sort=True).groups.values()]
draws=[np.concatenate([groups[g] for g in rng.integers(0,len(groups),len(groups))]) for _ in range(2000)]
longrows=[]
for name in ['locked8','intercept_calibrated8','affine_calibrated8']:
    y=long.observed_neutrophils_percent.to_numpy(float);p=long[name].to_numpy(float)
    row=dict(model=name,n_visits=len(long),n_subjects=len(groups),**metrics(y,p))
    boot=[metrics(y[i],p[i]) for i in draws]
    for key in ['R2','MAE']:
        row[key+'_cluster_CI_low'],row[key+'_cluster_CI_high']=np.quantile([b[key] for b in boot],[.025,.975])
    yc=y-long.groupby('subject').observed_neutrophils_percent.transform('mean').to_numpy()
    pc=p-long.groupby('subject')[name].transform('mean').to_numpy()
    row['within_donor_R2']=float(1-np.sum((yc-pc)**2)/np.sum(yc**2))
    row['within_donor_Pearson_r']=float(np.corrcoef(yc,pc)[0,1])
    longrows.append(row)
pd.DataFrame(longrows).to_csv(OUT/'longitudinal_cluster_diagnostics.csv',index=False)

# Age-stratum diagnostics of the unadapted predictor, no new fitting.
strata=[]
for label,mask in [('age_below45',first.age.lt(45)),('age_at_least45',first.age.ge(45))]:
    sub=first[mask];strata.append(dict(stratum=label,n=len(sub),**metrics(sub.observed_neutrophils_percent.to_numpy(),sub.locked8.to_numpy())))
pd.DataFrame(strata).to_csv(OUT/'age_stratum_diagnostics.csv',index=False)

primary=next(r for r in rows if r['scope']=='external_primary_all94' and r['model']=='locked8')
gate=PROTOCOL['external_pilot_gate']
passed=(primary['R2']>=gate['R2_at_least'] and primary['MAE']<=gate['MAE_at_most_percent_points'] and primary['Spearman_r']>=gate['Spearman_at_least'] and contrasts[0]['CI_low']>0)
manifest=dict(protocol_sha256=sha(OUT/'execution_protocol.json'),source_model_sha256=sha(modelpath),
              independent_external_study=True,primary_subjects=len(first),heldout_calibration_subjects=len(held),
              clinical_target='Measured percent neutrophils',RNA_input='Ensembl91 exon-union reconstructed TPM from published HTSeq gene counts',
              primary_result=primary,external_exploratory_gate_passed=bool(passed),primary_MAE_increment=contrasts[0],
              source_constant=source_mean,source_gene_coverage=len(available)/len(genes),
              unclipped_predictions_outside_0_100=int(((unclipped<0)|(unclipped>100)).sum()),
              calibration_parameters=parameters,target_outcome_used_for_zero_shot_fit=False,
              target_site_calibration_is_supervised=True,calibration_donors_excluded_from_longitudinal_holdout=True,
              algorithm_superiority_established=False,AD_triage_benefit_established=False,
              preserved_before=preserved,preserved_after={n:sha(DATA/n) for n in preserved},
              software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,h5py=h5py.__version__),
              input_sha256={n:sha(OUT/n) for n in ['raw/sound-life_whole-blood_no-stim.h5ad','raw/sound_life_labs_metadata.csv','raw/Homo_sapiens.GRCh38.91.gtf.gz']})
assert manifest['preserved_before']==manifest['preserved_after']
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('External gate',bool(passed),'preserved original and locked model',flush=True)
