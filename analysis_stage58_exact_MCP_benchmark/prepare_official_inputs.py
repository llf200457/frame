import hashlib
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
DATA = OUT.parent
A = DATA / 'analysis_stage57_external_resource_audit'
S = DATA / 'analysis_stage53_measured_blood_anchor'
E = DATA / 'analysis_stage54_soundlife_external'
P = json.loads((OUT / 'execution_protocol.json').read_text(encoding='utf-8'))
for path, expected in P['preserve_files'].items():
    assert hashlib.sha256((DATA / path).read_bytes()).hexdigest() == expected
sets = pd.read_csv(A / 'official_MCPcounter_common_genes.tsv', sep='\t')
genes = sorted(set(sets['HUGO symbols']))
meta = pd.read_csv(S / 'paired_clinical_metadata.csv')
expr = pd.read_csv(S / 'raw/GSE157103_genes.tpm.tsv.gz', sep='\t', index_col=0)
np.log2(expr.loc[genes, meta['sample']] + 1).to_csv(OUT / 'source_official_marker_logTPM.csv', index_label='gene')
target = pd.read_csv(E / 'primary_one_visit_metadata.csv')
annotation = pd.read_csv(E / 'gene_length_annotation.csv')
length = annotation.exon_union_length_bp.to_numpy(float)
ix = {g: i for i, g in enumerate(annotation['name'])}
gene_indices = np.array([ix[g] for g in genes])
values = np.zeros((len(genes), len(target)))
with h5py.File(E / 'raw/sound-life_whole-blood_no-stim.h5ad') as h:
    assert np.array_equal(h['var/name'].asstr()[:], annotation['name'].to_numpy())
    for i, row in enumerate(target.itertuples()):
        rates = h['X'][int(row.h5_row), :].astype(float) / length
        values[:, i] = np.log2(rates[gene_indices] / rates.sum() * 1e6 + 1)
pd.DataFrame(values, index=genes, columns=target['sample']).to_csv(OUT / 'target_official_marker_logTPM.csv', index_label='gene')
manifest = dict(source_n=125, target_n=94, common_gene_n=len(genes), population_gene_entries=len(sets),
                gene_support='same common-source-target genes; all official10 scores to be computed',
                no_prediction_scoring=True, official_commit=json.loads((A / 'download_manifest.json').read_text())['mcp_upstream_ref'])
(OUT / 'input_preparation_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(manifest, flush=True)
