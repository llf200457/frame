"""Lock eligibility and sample mapping for GSE249477 before model scoring."""
from pathlib import Path
import csv
import gzip
import json
import re
from collections import Counter
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE if (BASE / 'modules').is_dir() else BASE.parents[1]
OUT = ROOT / 'analysis_stage27'
series = ROOT / 'raw/GSE249477/GSE249477_series_matrix.txt.gz'
matrix = ROOT / 'raw/GSE249477/GSE249477_raw_count_normalize_04-10-2025.csv.gz'
fields = {}
with gzip.open(series, 'rt', encoding='utf-8', errors='replace') as fh:
    for line in fh:
        if line.startswith('!series_matrix_table_begin'):
            break
        if line.startswith('!Sample_'):
            p = next(csv.reader([line], delimiter='\t'))
            fields.setdefault(p[0], []).append(p[1:])
n = len(fields['!Sample_geo_accession'][0])
assert all(len(v) == n for groups in fields.values() for v in groups)
records = []
for i in range(n):
    item = {'geo_accession': fields['!Sample_geo_accession'][0][i],
            'title': fields['!Sample_title'][0][i]}
    for group in fields['!Sample_characteristics_ch1']:
        if ':' in group[i]:
            key, value = group[i].split(':', 1)
            item[key.lower().strip().replace(' ', '_')] = value.strip()
    item['sample_id'] = re.search(r'\[(DK\d+_\d+)\s*\]', item['title']).group(1)
    records.append(item)
meta = pd.DataFrame(records)
assert meta.sample_id.is_unique and meta.geo_accession.is_unique
meta['status'] = meta['disease_state'].map({
    "Alzheimer's disease": 'AD',
    "mild cognitive impairment due to Alzheimer's disease": 'MCI',
    'healthy control': 'CTL',
    'healthy controls': 'CTL',
    'healthy': 'CTL',
    'cognitively normal control': 'CTL',
})
header = next(csv.reader(gzip.open(matrix, 'rt', encoding='utf-8', errors='replace')))
tpm = [x for x in header if x.endswith(' - TPM')]
matrix_ids = [x.split(' (GE)')[0] for x in tpm]
meta['matrix_available'] = meta.sample_id.isin(matrix_ids)
assert len(matrix_ids) == len(set(matrix_ids)) == 62
assert set(matrix_ids) == set(meta.loc[meta.matrix_available, 'sample_id'])
assert meta.loc[meta.matrix_available, 'status'].notna().all(), meta.disease_state.value_counts().to_dict()
frozen = pd.read_csv(ROOT / 'modules/frozen_pca_modules.csv').gene
expr = pd.read_csv(matrix, usecols=['Name'] + tpm, compression='gzip', low_memory=False)
expr = expr.dropna(subset=['Name'])
expr['Name'] = expr['Name'].astype(str)
valid_symbol = expr.Name.str.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
complete_values = expr[tpm].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
malformed = expr.loc[~(valid_symbol & complete_values),'Name'].tolist()
non_symbol_ids = int(expr.Name.str.startswith('gene:ENSG').sum())
incomplete_rows = int((~complete_values).sum())
expr = expr.loc[valid_symbol & complete_values]
mapped = set(expr.Name)
coverage = len(set(frozen) & mapped)
min_ad_ctl = min(meta.loc[meta.matrix_available,'status'].value_counts().get('AD',0),
                 meta.loc[meta.matrix_available,'status'].value_counts().get('CTL',0))
criteria = {
    'separate_GEO_study': True,
    'whole_blood': True,
    'individual_expression_matrix': len(tpm) == 62,
    'unambiguous_labels': bool(meta.loc[meta.matrix_available,'status'].notna().all()),
    'at_least_20_per_AD_CTL': bool(min_ad_ctl >= 20),
    'at_least_80_percent_frozen_genes': bool(coverage >= 4000),
}
audit = {
    'geo_metadata_samples': n,
    'matrix_samples': len(tpm),
    'metadata_label_counts': meta.status.value_counts().to_dict(),
    'matrix_label_counts': meta.loc[meta.matrix_available,'status'].value_counts().to_dict(),
    'unmapped_metadata_ids': meta.loc[~meta.matrix_available,'sample_id'].tolist(),
    'age_by_status': {k: {'min': int(g.age.str.extract(r'(\d+)')[0].astype(int).min()),
                          'median': float(g.age.str.extract(r'(\d+)')[0].astype(int).median()),
                          'max': int(g.age.str.extract(r'(\d+)')[0].astype(int).max())}
                      for k,g in meta.loc[meta.matrix_available].groupby('status')},
    'frozen_gene_coverage': coverage,
    'frozen_gene_coverage_fraction': coverage / 5000,
    'duplicate_gene_symbol_rows': int(expr.Name.duplicated().sum()),
    'non_symbol_gene_identifiers_skipped': non_symbol_ids,
    'incomplete_or_malformed_numeric_rows_skipped': incomplete_rows,
    'all_rows_excluded_from_symbol_mapping': len(malformed),
    'frozen_genes_in_excluded_rows': sorted(set(frozen) & set(malformed)),
    'eligibility_criteria_locked_before_scoring': criteria,
    'eligible': all(criteria.values()),
}
meta.to_csv(OUT / 'GSE249477_linked_metadata.csv', index=False)
(OUT / 'eligibility_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps(audit, indent=2))
