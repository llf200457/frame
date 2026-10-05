"""Freeze source-only baselines, then reconstruct fixed external marker inputs."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
threadpool_limits(limits=1)
OUT = Path(__file__).resolve().parent
DATA = OUT.parent
S = DATA / 'analysis_stage53_measured_blood_anchor'
E = DATA / 'analysis_stage54_soundlife_external'
P = json.loads((OUT / 'execution_protocol.json').read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def decode(node):
    if isinstance(node, h5py.Group):
        categories = decode(node['categories'])
        codes = node['codes'][:]
        return np.array([categories[c] if c >= 0 else '' for c in codes])
    return node.asstr()[:] if node.dtype.kind in 'OS' else node[:]


preserved = {n: sha(DATA / n) for n in P['preserve_files']}
assert preserved == P['preserve_files'], 'A frozen dependency has changed: audit before proceeding'
coverage = pd.read_csv(S / 'marker_gene_coverage.csv')
signatures = {r.cell: r.genes.split(';') for r in coverage.itertuples()}
genes = sorted({g for v in signatures.values() for g in v})
assert len(genes) == 61 and len(signatures) == 7
source = pd.read_csv(S / 'paired_clinical_metadata.csv')
source_tpm = pd.read_csv(S / 'raw/GSE157103_genes.tpm.tsv.gz', sep='\t', index_col=0)
assert len(source) == 125 and source['sample'].is_unique and source_tpm.index.is_unique
assert set(genes) <= set(source_tpm.index)
source_cov = source[['age', 'male', 'sex_missing']].to_numpy(float)
source_log = np.log2(source_tpm.loc[genes, source['sample']].astype(float) + 1)
source_scores = pd.DataFrame({cell: source_log.loc[gs].mean(axis=0) for cell, gs in signatures.items()})
source_scores = source_scores.loc[source['sample']]
source_features = np.column_stack([source_cov, source_scores.to_numpy(float)])
source_y = source.target_neutrophils_percent.to_numpy(float)
frozen = {'training_n': 125, 'target': 'Measured neutrophils percent', 'alpha': 10.0,
          'source_target_mean': float(source_y.mean()), 'marker_gene_sets': signatures,
          'source_sample_ids': source['sample'].tolist(), 'models': {},
          'feature_scaling': 'Source125 StandardScaler only; marker scores are mean log2(TPM+1)',
          'protocol_sha256': sha(OUT / 'execution_protocol.json'),
          'source_dependencies_sha256': {str(p.relative_to(DATA)): sha(p) for p in [
              S / 'paired_clinical_metadata.csv', S / 'raw/GSE157103_genes.tpm.tsv.gz', S / 'marker_gene_coverage.csv']}}
fit_checks = []
for name, x, feature_names in [
    ('age_sex', source_cov, ['age', 'male', 'sex_missing']),
    ('marker7', source_features, ['age', 'male', 'sex_missing'] + list(signatures))]:
    scale = StandardScaler().fit(x)
    z = scale.transform(x)
    model = Ridge(alpha=10.0).fit(z, source_y)
    # Independent closed-form ridge check; scaled source columns have zero means.
    z_centered = z - z.mean(axis=0)
    coef_formula = np.linalg.solve(z_centered.T @ z_centered + 10.0 * np.eye(z.shape[1]),
                                   z_centered.T @ (source_y - source_y.mean()))
    fit_checks.append({'model': name, 'closed_form_max_coefficient_difference': float(np.max(np.abs(coef_formula - model.coef_)))})
    assert np.allclose(coef_formula, model.coef_, atol=1e-10, rtol=0)
    frozen['models'][name] = dict(feature_names=feature_names, mean=scale.mean_.tolist(),
                                 scale=scale.scale_.tolist(), coef=model.coef_.tolist(), intercept=float(model.intercept_))
# These artifacts are written BEFORE reading/calibrating/scoring any external outcomes.
model_path = OUT / 'source_baseline_parameters.json'
model_path.write_text(json.dumps(frozen, indent=2), encoding='utf-8')
pd.DataFrame(fit_checks).to_csv(OUT / 'source_closed_form_fit_checks.csv', index=False)
source_scores.to_csv(OUT / 'source_marker_scores.csv', index_label='sample')
print('Source-only baselines frozen:', sha(model_path), flush=True)

primary = pd.read_csv(E / 'primary_one_visit_metadata.csv')
assert len(primary) == 94 and primary.subject.is_unique and primary['sample'].is_unique
annotation = pd.read_csv(E / 'gene_length_annotation.csv')
lengths = annotation.exon_union_length_bp.to_numpy(float)
assert len(lengths) == 58302 and np.all(np.isfinite(lengths)) and np.all(lengths > 0)
target_marker = np.zeros((len(genes), len(primary)), dtype=float)
overlap_checks, pairing_checks = [], []
previous = pd.read_pickle(E / 'SoundLife_model_gene_TPM.pkl', compression='gzip')
with h5py.File(E / 'raw/sound-life_whole-blood_no-stim.h5ad', 'r') as h:
    var_names, var_ids = decode(h['var/name']), decode(h['var/id'])
    assert np.array_equal(var_names, annotation['name'].to_numpy())
    assert np.array_equal(var_ids, annotation['id'].to_numpy())
    assert len(set(var_names)) == len(var_names)
    assert set(genes) <= set(var_names), 'Do not change marker list after scoring'
    positions = {g: i for i, g in enumerate(var_names)}
    marker_indices = np.array([positions[g] for g in genes])
    shared = [g for g in genes if g in previous.index]
    shared_indices = np.array([positions[g] for g in shared])
    obs_kit, obs_subject, obs_index = (decode(h['obs/' + field]) for field in
                                     ['sample.sampleKitGuid', 'subject.subjectGuid', '_index'])
    for j, row in enumerate(primary.itertuples(index=False)):
        hrow = int(row.h5_row)
        assert obs_kit[hrow] == row.sample and obs_subject[hrow] == row.subject
        rates = h['X'][hrow, :].astype(float) / lengths
        denominator = rates.sum()
        assert denominator > 0 and np.all(rates >= 0)
        tpm = rates / denominator * 1e6
        target_marker[:, j] = tpm[marker_indices]
        old = previous.loc[shared, row.sample].to_numpy(float)
        abs_diff = float(np.max(np.abs(old - tpm[shared_indices])))
        overlap_checks.append(dict(sample=row.sample, shared_marker_gene_n=len(shared),
                                   max_abs_TPM_difference=abs_diff, full_TPM_sum=float(tpm.sum()),
                                   full_gene_denominator=denominator))
        assert abs_diff < 1e-8 and abs(tpm.sum() - 1e6) < 1e-8
        pairing_checks.append(dict(sample=row.sample, subject=row.subject, h5_row=hrow,
                                   raw_library=obs_index[hrow], exact_kit_match=True, exact_subject_match=True))
target_tpm = pd.DataFrame(target_marker, index=genes, columns=primary['sample'])
target_tpm.to_csv(OUT / 'target_marker_TPM.csv', index_label='gene')
target_log = np.log2(target_tpm + 1)
target_scores = pd.DataFrame({cell: target_log.loc[gs].mean(axis=0) for cell, gs in signatures.items()})
target_scores.to_csv(OUT / 'target_marker_scores.csv', index_label='sample')
pd.DataFrame(overlap_checks).to_csv(OUT / 'TPM_overlap_checks.csv', index=False)
pd.DataFrame(pairing_checks).to_csv(OUT / 'raw_pairing_checks.csv', index=False)
pd.DataFrame([dict(cell=cell, fixed_gene_n=len(gs), source_available=len(gs), target_available=len(gs),
                   target_imputed_gene_n=0, genes=';'.join(gs)) for cell, gs in signatures.items()]).to_csv(
                       OUT / 'marker_availability.csv', index=False)
manifest = dict(status='Source-only baselines frozen and metadata-only external input prepared; no new external scores yet',
                source_model_sha256=sha(model_path), protocol_sha256=sha(OUT / 'execution_protocol.json'),
                source_n=125, primary_target_subject_n=94, exact_shared_marker_gene_n=61,
                previous_TPM_shared_marker_n=len(shared), max_TPM_difference=max(r['max_abs_TPM_difference'] for r in overlap_checks),
                source_fit_checks=fit_checks, preserved_before=preserved,
                preserved_after={n: sha(DATA / n) for n in preserved},
                inputs_sha256={str(p.relative_to(DATA)): sha(p) for p in [
                    E / 'raw/sound-life_whole-blood_no-stim.h5ad', E / 'gene_length_annotation.csv',
                    E / 'primary_one_visit_metadata.csv', E / 'fixed_calibration_subjects.csv']},
                software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                              h5py=h5py.__version__, sklearn=sklearn.__version__))
assert manifest['preserved_before'] == manifest['preserved_after']
(OUT / 'preparation_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Marker preparation complete; all61 available; max TPM difference:', manifest['max_TPM_difference'], flush=True)
