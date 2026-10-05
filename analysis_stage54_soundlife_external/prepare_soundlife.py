"""Exact sample-kit join and documented gene-length TPM reconstruction.

No predictor coefficients or target-outcome associations are inspected here.
"""
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
RAW=OUT/'raw'
PROTOCOL=json.loads((OUT/'execution_protocol.json').read_text(encoding='utf-8'))
modelpath=OUT.parent/'analysis_stage53_measured_blood_anchor/locked_neutrophil_factor_predictor.npz'
assert hashlib.sha256(modelpath.read_bytes()).hexdigest()==PROTOCOL['expected_model_sha256']
model=np.load(modelpath,allow_pickle=False)


def h5value(q):
    if isinstance(q,h5py.Group):
        cats=h5value(q['categories']);codes=q['codes'][:]
        return np.array([cats[i] if i>=0 else None for i in codes])
    return q.asstr()[:] if q.dtype.kind in ['O','S'] else q[:]


intervals=defaultdict(list)
with gzip.open(RAW/'Homo_sapiens.GRCh38.91.gtf.gz','rt',encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'):continue
        v=line.rstrip().split('\t')
        if len(v)!=9 or v[2]!='exon':continue
        gene=re.search(r'gene_id "([^"]+)"',v[8]).group(1).split('.')[0]
        intervals[(gene,v[0],v[6])].append((int(v[3]),int(v[4])))
lengths=defaultdict(int)
for (gene,chrom,strand),parts in intervals.items():
    parts.sort();lo,hi=parts[0];total=0
    for a,b in parts[1:]:
        if a>hi+1:total+=hi-lo+1;lo,hi=a,b
        else:hi=max(hi,b)
    lengths[gene]+=total+hi-lo+1
print('Official exon-union gene lengths',len(lengths),flush=True)

with h5py.File(RAW/'sound-life_whole-blood_no-stim.h5ad','r') as f:
    obs=pd.DataFrame({k:h5value(f['obs'][k]) for k in f['obs']})
    var=pd.DataFrame({k:h5value(f['var'][k]) for k in f['var']})
    assert var.id.is_unique and var['name'].is_unique and obs['_index'].is_unique
    assert f['X'].shape==(len(obs),len(var)) and f['X'].dtype.kind=='i'
    obs['h5_row']=np.arange(len(obs))
    obs['library_number']=obs['_index'].str.extract(r'(\d+)$')[0].astype(int)
    duplicate=obs[obs['sample.sampleKitGuid'].duplicated(False)].copy()
    duplicate.to_csv(OUT/'technical_duplicate_audit.csv',index=False)
    obs=obs.sort_values('library_number').drop_duplicates('sample.sampleKitGuid',keep='last').sort_values('h5_row')
    clinical=pd.read_csv(RAW/'sound_life_labs_metadata.csv')
    assert clinical['sample.sampleKitGuid'].is_unique
    paired=obs.merge(clinical,on='sample.sampleKitGuid',how='left',validate='one_to_one',suffixes=('_RNA','_CBC'),indicator=True)
    assert paired['_merge'].eq('both').all()
    for key in ['subject.subjectGuid','subject.biologicalSex','sample.subjectAgeAtDraw','sample.daysSinceFirstVisit']:
        assert paired[key+'_RNA'].eq(paired[key+'_CBC']).all(),key
    paired['subject']=paired['subject.subjectGuid_RNA']
    paired['sample']=paired['sample.sampleKitGuid']
    paired['age']=pd.to_numeric(paired['sample.subjectAgeAtDraw_RNA'],errors='raise')
    paired['male']=paired['subject.biologicalSex_RNA'].map({'Male':1.0,'Female':0.0})
    assert paired.male.notna().all()
    paired['sex_missing']=0.0
    paired['target_neutrophils_percent']=pd.to_numeric(paired['bc.perc_neutrophils'],errors='coerce')
    excluded=paired[paired.target_neutrophils_percent.isna()].copy()
    excluded.to_csv(OUT/'missing_CBC_exclusions.csv',index=False)
    paired=paired[paired.target_neutrophils_percent.notna()].copy().reset_index(drop=True)
    assert paired.target_neutrophils_percent.between(0,100).all()
    assert paired['sample'].is_unique
    primary=paired.sort_values(['sample.daysSinceFirstVisit_RNA','sample.sampleKitGuid','library_number']).drop_duplicates('subject',keep='first')
    primary=primary.sort_values('subject').reset_index(drop=True)
    assert primary.subject.is_unique and len(primary)==paired.subject.nunique()
    # Reserve fixed subjects before any model predictions are scored.
    rng=np.random.default_rng(PROTOCOL['seed'])
    calibration_subjects=set(primary.subject.to_numpy()[rng.permutation(len(primary))[:20]])
    paired['role']=np.where(paired.subject.isin(calibration_subjects),'calibration_subject','heldout_subject')
    primary['role']=np.where(primary.subject.isin(calibration_subjects),'calibration_subject','heldout_subject')
    primary['primary_one_visit']=True
    paired['primary_one_visit']=paired['sample'].isin(primary['sample'])
    primary.to_csv(OUT/'primary_one_visit_metadata.csv',index=False)
    paired.to_csv(OUT/'all_paired_visit_metadata.csv',index=False)
    pd.DataFrame({'subject':sorted(calibration_subjects)}).to_csv(OUT/'fixed_calibration_subjects.csv',index=False)

    var['exon_union_length_bp']=[lengths.get(g.split('.')[0],0) for g in var.id]
    var.to_csv(OUT/'gene_length_annotation.csv',index=False)
    length=var.exon_union_length_bp.to_numpy(float)
    has_length=length>0
    modelgenes=model['genes'].tolist()
    lookup=dict(zip(var['name'],np.arange(len(var))))
    found=[g for g in modelgenes if g in lookup and length[lookup[g]]>0]
    foundrows=np.array([lookup[g] for g in found])
    output=np.zeros((len(paired),len(found)))
    audits=[]
    for start in range(0,len(paired),48):
        take=paired.h5_row.to_numpy(int)[start:start+48]
        # h5py fancy indexing requires sorted unique row indices; paired retained order.
        assert np.all(np.diff(take)>0)
        counts=f['X'][take,:].astype(float)
        assert np.isfinite(counts).all() and (counts>=0).all()
        rates=np.zeros_like(counts)
        rates[:,has_length]=counts[:,has_length]/length[has_length]
        denominator=rates.sum(axis=1)
        assert (denominator>0).all()
        output[start:start+len(take)]=1e6*rates[:,foundrows]/denominator[:,None]
        for j,row in enumerate(take):
            audits.append(dict(h5_row=int(row),total_counts=float(counts[j].sum()),
                              unlengthable_counts=float(counts[j,~has_length].sum()),
                              unlengthable_count_fraction=float(counts[j,~has_length].sum()/counts[j].sum()),
                              reconstructed_TPM_sum=float((1e6*rates[j]/denominator[j]).sum())))
    pd.DataFrame(audits).to_csv(OUT/'TPM_reconstruction_audit.csv',index=False)
    expr=pd.DataFrame(output.T,index=found,columns=paired['sample'])
    expr.to_pickle(OUT/'SoundLife_model_gene_TPM.pkl',compression='gzip')
    pd.DataFrame({'gene':modelgenes,'available':[g in found for g in modelgenes]}).to_csv(OUT/'locked_predictor_gene_coverage.csv',index=False)
    manifest=dict(raw_libraries=int(f['X'].shape[0]),unique_RNA_sample_kits=len(obs),technical_duplicate_libraries_removed=int(f['X'].shape[0]-len(obs)),
                  clinical_rows=len(clinical),paired_visits=len(paired),missing_CBC_visits=len(excluded),primary_independent_subjects=len(primary),
                  calibration_subjects=len(calibration_subjects),heldout_subjects=len(primary)-len(calibration_subjects),
                  source_predictor_gene_n=len(modelgenes),source_predictor_genes_available=len(found),genes_without_exon_length=int((~has_length).sum()),
                  max_unlengthable_count_fraction=max(r['unlengthable_count_fraction'] for r in audits),
                  gene_reference='Ensembl GRCh38.91 exon unions; reconstructed TPM',exact_kit_subject_age_sex_match=True,
                  model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((OUT/'execution_protocol.json').read_bytes()).hexdigest())
    (OUT/'preparation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)
