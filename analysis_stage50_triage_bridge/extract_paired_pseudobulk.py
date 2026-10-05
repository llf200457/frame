"""Extract author raw_counts and real protein summaries, one row per donor.

Only selected rows of the sparse raw-count layer are fetched. X/raw.X are
log-normalized/scaled expressions and are intentionally NOT summed as counts.
"""
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from http_range_io import ZenodoRangeReader

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
obs = pd.read_csv(ROOT/'microglia_obs.csv', index_col=0, dtype={'SampleID':str})
genes = pd.read_csv(ROOT/'microglia_var.csv')['barcode'].astype(str).to_numpy()
assert len(set(genes))==len(genes)
paired = (obs.prep_type=='10x3prime_inCITE_soloTE') & np.isfinite(obs.total_counts_NFKB) & (obs.total_counts_H3>1)
assert obs.loc[paired,'SampleID'].nunique()==45
cell_counts = obs.loc[paired].groupby('SampleID').size()
eligible_ids = cell_counts[cell_counts>=20].index.sort_values()
included = paired & obs.SampleID.isin(eligible_ids)
rows = np.flatnonzero(included)
donor_index = {donor:i for i,donor in enumerate(eligible_ids)}
sums = np.zeros((len(eligible_ids),len(genes)),dtype=np.float64)
cell_totals = np.zeros(len(obs),dtype=np.float64)

record = json.loads((DATA/'analysis_stage49_top_journal_outlook/raw/omar_record.json').read_text(encoding='utf-8-sig'))
entry = next(x for x in record['files'] if x['key']=='Omar_et_al_MicrogliaData.h5ad')
url = 'https://zenodo.org/records/14712707/files/Omar_et_al_MicrogliaData.h5ad?download=1'
with ZenodoRangeReader(url,entry['size'],ROOT/'raw/microglia_range_cache',ROOT/'raw/Omar_et_al_MicrogliaData.h5ad.part') as reader:
    with h5py.File(reader,'r') as handle:
        group = handle['layers/raw_counts']
        assert str(group.attrs['encoding-type'])=='csr_matrix'
        pointers = group['indptr'][:]
        sample_values = group['data'][:100]
        assert np.all(sample_values>=0) and np.max(np.abs(sample_values-np.round(sample_values)))<1e-6
        for start in range(int(rows.min()),int(rows.max())+1,1500):
            stop=min(start+1500,len(obs))
            lo,hi=int(pointers[start]),int(pointers[stop])
            print(f'Extracting raw count cells {start}:{stop}; sparse entries {lo}:{hi}',flush=True)
            values=group['data'][lo:hi]
            indices=group['indices'][lo:hi]
            assert np.isfinite(values).all() and np.all(values>=0) and np.max(np.abs(values-np.round(values)))<1e-6
            matrix=csr_matrix((values,indices,pointers[start:stop+1]-lo),shape=(stop-start,len(genes)))
            selected_local=np.flatnonzero(included.iloc[start:stop].to_numpy())
            cell_totals[start:stop]=np.asarray(matrix.sum(axis=1)).ravel()
            donors=obs.iloc[start:stop].SampleID.to_numpy()
            for donor in set(donors[selected_local]):
                valid=selected_local[donors[selected_local]==donor]
                sums[donor_index[donor]] += np.asarray(matrix[valid].sum(axis=0)).ravel()

assert np.all(sums.sum(axis=1)>0)
expression=np.log2(1+1e6*sums/sums.sum(axis=1,keepdims=True))
np.savez_compressed(ROOT/'paired_microglia_pseudobulk.npz',donors=np.array(eligible_ids,dtype=str),genes=genes.astype(str),counts=sums,log2cpm=expression)
summary=[]
metadata = pd.read_csv(DATA/'analysis_stage49_top_journal_outlook/raw/omar_metadata.csv',dtype={'SampleID':str})
metadata=metadata.loc[metadata.Exclusion.fillna('').str.strip().eq('')].copy()
wb_fields=['pTau_>200_max','pTau_>200_mean','pTau_<200_max','pTau_<200_mean']
for field in wb_fields: metadata[field]=pd.to_numeric(metadata[field],errors='coerce')
for donor in eligible_ids:
    cells=obs.loc[included & obs.SampleID.eq(donor)]
    nf=cells.total_counts_NFKB.astype(float).to_numpy()
    h3=cells.total_counts_H3.astype(float).to_numpy()
    ratio=(nf+1)/h3
    discrepancy=float(np.max(np.abs(ratio-cells.total_counts_NFKB_norm_H3.to_numpy())))
    assert discrepancy<1e-5
    row={'donor':donor,'microglia_n':len(cells),'NFkB_log1p_ratio_mean':float(np.log1p(ratio).mean()),
         'NFkB_ratio_mean':float(ratio.mean()),'NFkB_ratio_median':float(np.median(ratio)),
         'NFkB_adt_mean':float(nf.mean()),'H3_adt_mean':float(h3.mean()),
         'n_counts_mean':float(cell_totals[obs.index.get_indexer(cells.index)].mean()),
         'n_genes_mean':float(cells.n_genes.mean()),'frac_mito_mean':float(cells.frac_mito.mean()),
         'RNA_library_sum':float(sums[donor_index[donor]].sum())}
    for field in ['Age','Sex','DiseaseState','PostmortemInterval','JoinedRIN','region','bank']:
        valid=cells[field].dropna().unique()
        if field in ['JoinedRIN','PostmortemInterval']:
            # Repeated aliquots can differ in technical annotations. Pool
            # within a donor, keep the range, and never split that person.
            numeric=pd.to_numeric(cells[field],errors='coerce')
            row[field]=float(numeric.mean()) if numeric.notna().any() else None
            row[field+'_source_nunique']=len(valid)
            row[field+'_source_range']=';'.join(str(v) for v in sorted(valid))
        else:
            assert len(valid)<=1,(donor,field,valid)
            row[field]=valid[0] if len(valid) else None
    source=metadata.loc[metadata.SampleID==donor]
    dates=source['date'].dropna().unique()
    row['metadata_batch']=dates[0] if len(dates)==1 else None
    row['metadata_batch_options']=';'.join(sorted(dates))
    row['source_metadata_rows']=len(source)
    row['has_unambiguous_source_record']=len(source)>0
    gels=source.Gel.dropna().unique()
    row['WB_gel']=gels[0] if len(gels)==1 else None
    for field in wb_fields:
        values=source[field].dropna().unique()
        assert len(values)<=1,(donor,field,values)
        row[field]=float(values[0]) if len(values) else None
    summary.append(row)
summary=pd.DataFrame(summary)
summary.to_csv(ROOT/'paired_donor_protein_clinical_wb.csv',index=False,encoding='utf-8-sig')
cell_counts.rename('qualified_paired_microglia_n').to_frame().assign(included=cell_counts.index.isin(eligible_ids)).to_csv(ROOT/'paired_donor_inclusion.csv')

# Gene links to the actual original blood modules, no additional gene screening.
modules=pd.read_csv(DATA/'modules/frozen_pca_modules.csv')
coverage=[]
for module,part in modules.groupby('module'):
    measured=part.gene.isin(genes)
    coverage.append({'module':module,'frozen_blood_gene_n':len(part),'present_in_microglia_raw_count_matrix':int(measured.sum()),'coverage':float(measured.mean())})
pd.DataFrame(coverage).to_csv(ROOT/'blood_module_microglia_gene_coverage.csv',index=False)

# Read Excel XML as source data without installing spreadsheet packages.
with zipfile.ZipFile(ROOT/'raw/pTau_Blots.zip') as archive:
    workbook=archive.read('pTau_Blots/WesternData_pTauLadder.xlsx')
with zipfile.ZipFile(__import__('io').BytesIO(workbook)) as z:
    sheets=ET.fromstring(z.read('xl/workbook.xml'))
    namespace={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    names=[s.attrib['name'] for s in sheets.findall('s:sheets/s:sheet',namespace)]
    (ROOT/'wb_workbook_sheet_inventory.json').write_text(json.dumps(names,ensure_ascii=False,indent=2),encoding='utf-8')

manifest={'date':'2026-10-02','author_processed_microglia_cells':len(obs),'author_all_rna_donors':int(obs.SampleID.nunique()),
          'inCITE_cells':int((obs.prep_type=='10x3prime_inCITE_soloTE').sum()),'paired_H3_gt1_cells':int(paired.sum()),
          'paired_donors_before_minimum_cell_gate':45,'minimum_paired_microglia_cells_per_donor':20,
          'included_donors':len(eligible_ids),'included_cells':int(included.sum()),
          'included_disease_counts':summary.DiseaseState.value_counts().to_dict(),
          'WB_any_field_complete_donors':int(summary[wb_fields].notna().any(axis=1).sum()),
          'batch_assignment_unique_donors':int(summary.metadata_batch.notna().sum()),
          'donors_with_multiple_RIN_values':summary.loc[summary.JoinedRIN_source_nunique>1,'donor'].tolist(),
          'repeated_aliquot_technical_covariate_aggregation':'cell-count-weighted within donor; source ranges retained; no donor split',
          'used_gene_matrix':'layers/raw_counts; nonnegative integer-valued checked in every extracted block',
          'did_not_sum_X_or_log_normalized_raw_X':True,'protein_target':'mean of per-nucleus log1p((NFkB_ADT+1)/H3_ADT)',
          'primary_WB_field_not_yet_reconciled':True,'whole_file_MD5_verified':False,'range_blocks_content_header_verified':True,
          'RNA_protein_cohort_model_not_yet_trained':True,
          'outputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'paired_microglia_pseudobulk.npz',ROOT/'paired_donor_protein_clinical_wb.csv']}}
(ROOT/'paired_pseudobulk_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
