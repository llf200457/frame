"""Audit GEO sample labels, matrix linkage, and frozen-gene coverage."""
from pathlib import Path
import csv
import gzip
import json
import re
from collections import Counter
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE if (BASE/'modules').is_dir() else BASE.parents[1]
OUT = ROOT / 'analysis_stage26'

def sample_metadata(accession):
    path = ROOT / 'raw' / accession / f'{accession}_series_matrix.txt.gz'
    fields = {}
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('!series_matrix_table_begin'):
                break
            if not line.startswith('!Sample_'):
                continue
            parts = next(csv.reader([line], delimiter='\t'))
            fields.setdefault(parts[0], []).append(parts[1:])
    count = len(fields['!Sample_geo_accession'][0])
    for key, groups in fields.items():
        assert all(len(group) == count for group in groups), (accession, key)
    rows = []
    for i in range(count):
        row = {'geo_accession': fields['!Sample_geo_accession'][0][i],
               'title': fields['!Sample_title'][0][i],
               'source': fields['!Sample_source_name_ch1'][0][i],
               'platform': fields['!Sample_platform_id'][0][i]}
        for group in fields.get('!Sample_characteristics_ch1', []):
            item = group[i]
            if ':' in item:
                key, value = item.split(':', 1)
                row[key.strip().lower().replace(' ', '_')] = value.strip()
        rows.append(row)
    return pd.DataFrame(rows)

def write_metadata(accession, frame):
    frame.to_csv(OUT / f'{accession}_metadata.csv', index=False)
    print(accession, 'GEO metadata', len(frame), 'columns', frame.columns.tolist())
    for column in ('diagnosis','age','sex','gender'):
        if column in frame:
            print(column, frame[column].value_counts(dropna=False).head(15).to_dict())

external = sample_metadata('GSE140829')
write_metadata('GSE140829', external)
expr = pd.read_csv(ROOT/'raw/GSE140829/GSE140829_final_normalized_data.txt.gz',
                   sep='\t', index_col=0, compression='gzip')
external['expression_id'] = external['title'].str.extract(r',\s*([^,]+?)\s*\[ad_mci\]$')[0]
assert external['expression_id'].notna().all()
assert external['expression_id'].is_unique
assert expr.columns.is_unique and expr.index.is_unique
external['matrix_available'] = external['expression_id'].isin(expr.columns)
assert set(expr.columns) == set(external.loc[external.matrix_available,'expression_id'])
assert external.loc[external.matrix_available,'source'].eq('Whole blood').all()
write_metadata('GSE140829_linked', external)

tiny = sample_metadata('GSE97760')
write_metadata('GSE97760', tiny)
tiny['expression_id'] = tiny['title'].str.extract(r'^(.*?)\s*\(')[0]
tiny['matrix_column'] = tiny['title'].apply(
    lambda t: 'Norm_Control_'+re.search(r'Control\s*(\d+)',t).group(1)
    if 'Control' in t else 'Norm_'+re.search(r'(AP\d+)',t).group(1))
write_metadata('GSE97760_linked', tiny)

frozen = pd.read_csv(ROOT/'modules/frozen_pca_modules.csv')
discovery = pd.read_pickle(ROOT/'processed/GSE63060_gene_expression.pkl', compression='gzip')
target = pd.read_pickle(ROOT/'processed/GSE63061_gene_expression.pkl', compression='gzip')
extra = pd.read_pickle(ROOT/'processed/GSE85426_gene_expression.pkl', compression='gzip')
overlap = {'GSE140829_genes': len(expr.index),
           'GSE140829_frozen_genes': len(set(frozen.gene)&set(expr.index)),
           'GSE140829_common_all_frozen': len(set(frozen.gene)&set(expr.index)&set(discovery.index)&set(target.index)),
           'GSE85426_frozen_genes': len(set(frozen.gene)&set(extra.index)),
           'GSE140829_samples_matrix': len(expr.columns),
           'GSE140829_samples_metadata': len(external),
           'GSE140829_matrix_diagnoses':external.loc[external.matrix_available,'diagnosis'].value_counts().to_dict(),
           'GSE140829_missing_diagnoses':external.loc[~external.matrix_available,'diagnosis'].value_counts().to_dict(),
           'GSE140829_sample_id_overlap_discovery':len(set(expr.columns)&set(discovery.columns)),
           'GSE140829_sample_id_overlap_target':len(set(expr.columns)&set(target.columns)),
           'GSE97760_metadata_n':len(tiny)}
(OUT/'eligibility_audit.json').write_text(json.dumps(overlap,indent=2), encoding='utf-8')
print(json.dumps(overlap,indent=2))
