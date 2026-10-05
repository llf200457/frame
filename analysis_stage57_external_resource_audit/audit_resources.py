"""Eligibility and signature audit only; no predictive result on candidate datasets."""
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
DATA = OUT.parent
RAW = OUT / 'raw'
rows = []
for path in sorted(RAW.glob('SRP429744_biosamples_*.xml')):
    for s in ET.parse(path).getroot().findall('SAMPLE'):
        record = dict(sample_accession=s.attrib['accession'], sample_alias=s.attrib.get('alias'),
                      sample_title=s.findtext('TITLE'), description=s.findtext('DESCRIPTION'))
        for a in s.findall('./SAMPLE_ATTRIBUTES/SAMPLE_ATTRIBUTE'):
            tag, value = a.findtext('TAG'), a.findtext('VALUE')
            assert tag not in record
            record[tag] = value
        rows.append(record)
meta = pd.DataFrame(rows)
runs = pd.read_csv(RAW / 'SRP429744_ena_runs.tsv', sep='\t')
assert len(meta) == 138 and meta.sample_accession.is_unique and meta.sample_alias.is_unique
assert len(runs) == 138 and runs.sample_accession.is_unique
assert set(meta.sample_accession) == set(runs.sample_accession)
joined = runs.merge(meta, on=['sample_accession', 'sample_alias'], validate='one_to_one')
joined['age_as_published'] = joined['age']
joined['age_at_least90_censored'] = joined['age'].eq('90+')
for col in ['age', 'body mass index', 'neutrophil count', 'lymphocyte count', 'monocyte count', 'eosinophil count', 'neutrophil to lymphocyte ratio']:
    joined[col] = pd.to_numeric(joined[col], errors='coerce')
assert joined['neutrophil count'].notna().all() and joined['neutrophil count'].between(0, 1).all()
assert (joined.age.notna() | joined.age_at_least90_censored).all()
assert joined.age.dropna().ge(18).all()
joined['age_lower_bound'] = joined.age.where(~joined.age_at_least90_censored, 90.0)
assert joined.sex.isin(['male', 'female']).all()
joined['measured_neutrophils_percent'] = 100 * joined['neutrophil count']
joined['reported_NLR_minus_fraction_ratio'] = joined['neutrophil to lymphocyte ratio'] - joined['neutrophil count'] / joined['lymphocyte count']
joined['four_cell_fraction_sum'] = joined[['neutrophil count', 'lymphocyte count', 'monocyte count', 'eosinophil count']].sum(axis=1)
joined.to_csv(OUT / 'SRP429744_matched_clinical_metadata.csv', index=False)
diagnoses = joined.disease.fillna('').str.lower().str.get_dummies(sep=';')
diagnosis_counts = pd.DataFrame({'recorded_disease_string': joined.disease.value_counts().index,
                                  'n': joined.disease.value_counts().values})
diagnosis_counts.to_csv(OUT / 'SRP429744_disease_string_counts.csv', index=False)
bytes_total = sum(int(w) for z in runs.fastq_bytes.astype(str) for w in z.split(';'))
entry = dict(accession='SRP429744 / PRJNA949611', measured_endpoint_verified=True,
             actual_downloaded_runs=138, actual_downloaded_sample_records=138, exact_one_to_one_ID_join=True,
             unique_sample_aliases=138, independent_donors_reported_by_author=138,
             sex_neutrophil_fraction_complete=True, exact_numeric_age_n=int(joined.age.notna().sum()),
             age_censored90plus_n=int(joined.age_at_least90_censored.sum()), age_range_numeric_uncensored=[float(joined.age.min()), float(joined.age.max())],
             age_at_least65_n=int(joined.age_lower_bound.ge(65).sum()), male_n=int(joined.sex.eq('male').sum()),
             neutrophils_percent_range=[float(joined.measured_neutrophils_percent.min()), float(joined.measured_neutrophils_percent.max())],
             measured_target_conversion='BioSample relative neutrophil count is a fraction of total leukocytes: multiply by100; not cells/uL',
             dementia_recorded_n=int(joined.disease.fillna('').str.contains('dementia', case=False).sum()),
             dementia_is_not_confirmed_AD=True, fastq_total_bytes=bytes_total, fastq_total_decimal_GB=bytes_total / 1e9,
             processed_expression_matrix_found=False, ENA_analysis_rows=len(pd.read_csv(RAW / 'SRP429744_ena_analysis.tsv', sep='\t')),
             raw_fastq_downloaded=False, predictive_scoring_executed=False,
             decision='Measured endpoint and matching verified; defer whole-cohort prediction pending feasible full gene quantification or an actually retrieved processed matrix')

# Official signatures versus previous hand-restricted marker means.
genes = pd.read_csv(RAW / 'MCPcounter_genes.txt', sep='\t')
source_genes = set(pd.read_csv(DATA / 'analysis_stage53_measured_blood_anchor/raw/GSE157103_genes.tpm.tsv.gz', sep='\t', index_col=0).index)
with h5py.File(DATA / 'analysis_stage54_soundlife_external/raw/sound-life_whole-blood_no-stim.h5ad') as h:
    target_genes = set(h['var/name'].asstr()[:])
prior = pd.read_csv(DATA / 'analysis_stage53_measured_blood_anchor/marker_gene_coverage.csv').set_index('cell')
coverage = []
for cell, group in genes.groupby('Cell population', sort=False):
    gs = set(group['HUGO symbols'])
    common = gs & source_genes & target_genes
    old = set(prior.loc[cell, 'genes'].split(';')) if cell in prior.index else set()
    coverage.append(dict(cell=cell, official_n=len(gs), source_available=len(gs & source_genes), target_available=len(gs & target_genes),
                         common_n=len(common), prior7_n=len(old), additional_common_genes=';'.join(sorted(common - old)),
                         source_missing=';'.join(sorted(gs - source_genes)), target_missing=';'.join(sorted(gs - target_genes)),
                         common_genes=';'.join(sorted(common))))
pd.DataFrame(coverage).to_csv(OUT / 'official_MCPcounter_marker_coverage.csv', index=False)
common_genes = genes[genes['HUGO symbols'].isin(source_genes & target_genes)]
common_genes.to_csv(OUT / 'official_MCPcounter_common_genes.tsv', sep='\t', index=False)
mcp = dict(commit=json.loads((OUT / 'download_manifest.json').read_text())['mcp_upstream_ref'], version='1.2.0',
           score_rule_verified='Official appendSignatures calculates mean expression of each intersected marker set per sample',
           prior7_not_full_official_signatures=True, prior_neutrophil_signature_exact=True,
           additional_common_immune_genes=['CD8B', 'KLRC4', 'IGKC', 'PAX5', 'KIR2DL3'],
           common_gene_population_entries=len(common_genes), code_sha256=hashlib.sha256((RAW / 'MCPcounter.R').read_bytes()).hexdigest(),
           raw_signatures_sha256=hashlib.sha256((RAW / 'MCPcounter_genes.txt').read_bytes()).hexdigest(),
           direct_cell_percentage_output=False, downstream_source_ridge_or_local_calibration_needed_for_absolute_target=True,
           reference='https://github.com/ebecht/MCPcounter')

# SLE backup: measured cell metadata exists, but arrays and repeated pediatric records.
sle = pd.read_csv(RAW / 'SLE_E-GEOD-65391.sdrf.txt', sep='\t')
neut = pd.to_numeric(sle['Characteristics [neutrophil_percent]'], errors='coerce')
age = pd.to_numeric(sle['Characteristics [age]'], errors='coerce')
eligible = neut.notna() & age.notna()
sle[['Source Name', 'Characteristics [subject]', 'Characteristics [age]', 'Characteristics [sex]',
     'Characteristics [neutrophil_count]', 'Characteristics [neutrophil_percent]']].to_csv(OUT / 'SLE_measured_endpoint_metadata.csv', index=False)
sle_entry = dict(accession='GSE65391 / E-GEOD-65391', downloaded_metadata_rows=len(sle), measured_neutrophil_percent_n=int(neut.notna().sum()),
                 both_age_percent_n=int(eligible.sum()), unique_subjects_with_age_percent=int(sle.loc[eligible, 'Characteristics [subject]'].nunique()),
                 age_range=[float(age[eligible].min()), float(age[eligible].max())],
                 percent_range=[float(neut.dropna().min()), float(neut.dropna().max())],
                 expression_type='whole-blood microarray, not TPM RNA-seq', expression_matrix_downloaded=False,
                 author_repository_expression_size_MB=429, repeated_records_not_independent_people=True,
                 prior_top_method_associations_not_out_of_sample_prediction=True, predictive_scoring_executed=False,
                 decision='Backup for future subject-disjoint array representation audit; do not insert values into frozen RNA-TPM predictor',
                 reference='https://shbrief.github.io/GenomicSuperSignaturePaper/articles/SLE-WB/neutrophil_counts_SLE-WB.html')

# Persist supplement descriptions to distinguish result summaries from expression matrices.
supplements = []
for path in RAW.glob('PMC*_full.xml'):
    root = ET.parse(path).getroot()
    for elem in root.findall('.//supplementary-material'):
        supplements.append(dict(paper=path.name, description=' '.join(' '.join(elem.itertext()).split()),
                                references=[dict(e.attrib) for e in elem.iter() if any(k.endswith('href') for k in e.attrib)]))
(OUT / 'published_supplement_index.json').write_text(json.dumps(supplements, indent=2), encoding='utf-8')
manifest = dict(date='2026-10-03', stage='metadata/source audit; candidate models not scored', new_resource=entry,
                mcp_audit=mcp, SLE_backup=sle_entry,
                network_failures_retained=['initial sandbox socket restriction', 'NCBI BioSample esearch free-text accession returned zero; ENA exact-accession retrieval succeeded',
                                           'Publisher requests returned small challenge pages, not full articles; EuropePMC XML retrieved for two papers', 'Genomics full XML endpoint returnedHTTP500'])
(OUT / 'resource_audit_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
