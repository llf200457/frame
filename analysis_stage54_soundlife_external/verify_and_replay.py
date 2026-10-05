"""Independent arithmetic checks, raw-count spot checks, and one full replay.

This script does not change endpoints, source weights, cohort roles or seeds.
Bootstrap intervals remain conditional on the fitted source and calibration sets.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
DATA = OUT.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


pipeline_csvs = [
    'technical_duplicate_audit.csv', 'missing_CBC_exclusions.csv',
    'primary_one_visit_metadata.csv', 'all_paired_visit_metadata.csv',
    'fixed_calibration_subjects.csv', 'gene_length_annotation.csv',
    'TPM_reconstruction_audit.csv', 'locked_predictor_gene_coverage.csv',
    'all_visit_predictions.csv', 'primary_one_visit_predictions.csv',
    'heldout_subject_predictions.csv', 'external_module_scores.csv',
    'external_prediction_performance.csv', 'paired_MAE_increments.csv',
    'longitudinal_cluster_diagnostics.csv', 'age_stratum_diagnostics.csv',
]
before = {name: sha(OUT / name) for name in pipeline_csvs}
old_prep = json.loads((OUT / 'preparation_manifest.json').read_text(encoding='utf-8'))
(OUT / 'preparation_manifest_before_final_protocol_replay.json').write_text(
    json.dumps(old_prep, indent=2), encoding='utf-8')

first = pd.read_csv(OUT / 'primary_one_visit_predictions.csv')
all_visits = pd.read_csv(OUT / 'all_visit_predictions.csv')
held = pd.read_csv(OUT / 'heldout_subject_predictions.csv')
performance = pd.read_csv(OUT / 'external_prediction_performance.csv')
parameters = json.loads((OUT / 'calibration_parameters.json').read_text(encoding='utf-8'))
metadata = pd.read_csv(OUT / 'all_paired_visit_metadata.csv')
model = np.load(DATA / 'analysis_stage53_measured_blood_anchor/locked_neutrophil_factor_predictor.npz', allow_pickle=False)

assert first.subject.is_unique and len(first) == 94
cal = first[first.role.eq('calibration_subject')]
assert len(cal) == 20 and len(held) == 74
assert set(cal.subject).isdisjoint(set(held.subject))
long = all_visits[all_visits.role.eq('heldout_subject')]
assert long.subject.nunique() == 74 and len(long) == 682
assert set(cal.subject).isdisjoint(set(long.subject))
assert set(first['sample']).issubset(set(all_visits['sample']))
assert all_visits['sample'].is_unique and len(all_visits) == 826

checks = []
for row in performance.to_dict('records'):
    frame = first if row['scope'] == 'external_primary_all94' else held
    y = frame.observed_neutrophils_percent.to_numpy(float)
    p = frame[row['model']].to_numpy(float)
    checked = {
        'R2': 1 - np.dot(y - p, y - p) / np.dot(y - y.mean(), y - y.mean()),
        'MAE': np.mean(np.abs(y - p)),
        'RMSE': np.sqrt(np.mean(np.square(y - p))),
        'bias': np.mean(p - y),
    }
    if np.ptp(p) > 0:
        checked['Spearman_r'] = np.corrcoef(rankdata(y), rankdata(p))[0, 1]
    for key, value in checked.items():
        error = abs(float(value) - row[key])
        assert error < 1e-10, (row['scope'], row['model'], key, error)
        checks.append({'scope': row['scope'], 'model': row['model'],
                       'metric': key, 'absolute_recalculation_error': error})
pd.DataFrame(checks).to_csv(OUT / 'independent_metric_checks.csv', index=False)

# Solve the calibration by closed-form arithmetic, independently of sklearn.
x = cal.locked8.to_numpy(float)
y = cal.observed_neutrophils_percent.to_numpy(float)
delta = float(np.mean(y - x))
slope = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
intercept = float(y.mean() - slope * x.mean())
assert abs(delta - parameters['intercept_delta']) < 1e-10
assert abs(slope - parameters['affine_slope']) < 1e-10
assert abs(intercept - parameters['affine_intercept']) < 1e-10
assert abs(y.mean() - parameters['target_calibration_mean']) < 1e-10

lengths = pd.read_csv(OUT / 'gene_length_annotation.csv')
coverage = pd.read_csv(OUT / 'locked_predictor_gene_coverage.csv')
audit = pd.read_csv(OUT / 'TPM_reconstruction_audit.csv')
assert len(lengths) == 58302 and lengths.exon_union_length_bp.gt(0).all()
assert np.abs(audit.reconstructed_TPM_sum - 1e6).max() < 1e-8
assert audit.unlengthable_count_fraction.eq(0).all()
genes = model['genes'].astype(str)
lookup = dict(zip(lengths['name'], range(len(lengths))))
available = [g for g in genes if g in lookup]
assert len(available) == 4705 and len(genes) == 4713
indices = np.array([i for i, g in enumerate(genes) if g in lookup])
count_indices = np.array([lookup[g] for g in available])
expr = pd.read_pickle(OUT / 'SoundLife_model_gene_TPM.pkl', compression='gzip')

# Cover a calibration visit, a held-out primary visit, a late visit, and the
# largest source prediction. The row identifiers are used, never row position.
samples = list(dict.fromkeys([
    cal.iloc[0]['sample'], held.iloc[0]['sample'],
    all_visits.iloc[-1]['sample'],
    all_visits.loc[all_visits.locked8.idxmax(), 'sample'],
]))
spot_checks = []
with h5py.File(OUT / 'raw/sound-life_whole-blood_no-stim.h5ad', 'r') as f:
    assert np.array_equal(f['var']['name'].asstr()[:], lengths['name'].to_numpy())
    assert np.array_equal(f['var']['id'].asstr()[:], lengths['id'].to_numpy())
    for sample in samples:
        meta = metadata[metadata['sample'].eq(sample)].iloc[0]
        saved = all_visits[all_visits['sample'].eq(sample)].iloc[0]
        counts = f['X'][int(meta.h5_row), :].astype(float)
        rates = counts / lengths.exon_union_length_bp.to_numpy(float)
        tpm = rates / rates.sum() * 1e6
        tpm_error = float(np.max(np.abs(tpm[count_indices] - expr.loc[available, sample].to_numpy(float))))
        z = np.zeros(len(genes))
        z[indices] = (np.log2(tpm[count_indices] + 1) - model['gene_mean'][indices]) / model['gene_sd'][indices]
        module = np.array([np.sum(z * model['module_weights'][:, j]) for j in range(8)])
        vector = np.concatenate(([meta.age, meta.male, meta.sex_missing], module))
        value = float(np.sum((vector - model['feature_mean']) / model['feature_scale'] * model['coefficients']) + model['intercept'])
        error = abs(value - saved.locked8_unclipped)
        assert error < 1e-9 and tpm_error < 1e-8
        spot_checks.append({'sample': sample, 'subject': meta.subject,
                            'raw_count_prediction_absolute_error': error,
                            'raw_count_TPM_max_absolute_error': tpm_error})
pd.DataFrame(spot_checks).to_csv(OUT / 'raw_count_prediction_spot_checks.csv', index=False)

# The physiological clipping was applied consistently. It explains why a
# constant intercept adjustment can slightly alter ranks and within-person r.
clipping = {}
for scope, frame in [('primary94', first), ('heldout74', held), ('heldout_visits682', long)]:
    p = frame.locked8.to_numpy(float)
    clipping[scope] = {
        'intercept_outside_0_100_before_clip': int(((p + delta < 0) | (p + delta > 100)).sum()),
        'affine_outside_0_100_before_clip': int(((intercept + slope * p < 0) | (intercept + slope * p > 100)).sum()),
    }

print('Independent metric, calibration, raw-count and subject-disjointness checks passed.', flush=True)
for script, log in [('prepare_soundlife.py', 'preparation_full_replay.log'),
                    ('run_external_validation.py', 'external_full_replay.log')]:
    print('Replaying', script, flush=True)
    with (OUT / log).open('w', encoding='utf-8') as stream:
        run = subprocess.run([sys.executable, '-u', str(OUT / script)],
                             cwd=OUT, stdout=stream, stderr=subprocess.STDOUT)
    assert run.returncode == 0, (script, run.returncode, log)
after = {name: sha(OUT / name) for name in pipeline_csvs}
table = [{'file': n, 'before_sha256': before[n], 'after_sha256': after[n],
          'byte_identical': before[n] == after[n]} for n in pipeline_csvs]
pd.DataFrame(table).to_csv(OUT / 'full_replay_checks.csv', index=False)
assert before == after
replayed_expr = pd.read_pickle(OUT / 'SoundLife_model_gene_TPM.pkl', compression='gzip')
assert expr.equals(replayed_expr)
prep = json.loads((OUT / 'preparation_manifest.json').read_text(encoding='utf-8'))
run = json.loads((OUT / 'run_manifest.json').read_text(encoding='utf-8'))
assert prep['protocol_sha256'] == run['protocol_sha256'] == sha(OUT / 'execution_protocol.json')
assert run['preserved_before'] == run['preserved_after']
for relative, expected in run['preserved_before'].items():
    assert sha(DATA / relative) == expected
result = {
    'metric_entries_checked': len(checks),
    'max_metric_absolute_error': max(v['absolute_recalculation_error'] for v in checks),
    'raw_count_samples_independently_checked': len(spot_checks),
    'max_raw_count_prediction_error': max(v['raw_count_prediction_absolute_error'] for v in spot_checks),
    'calibration_closed_form_verified': True,
    'calibration_and_heldout_donors_disjoint': True,
    'replayed_pipeline_csvs': len(pipeline_csvs),
    'all_pipeline_csvs_byte_identical': before == after,
    'reconstructed_TPM_matrix_numerically_identical': expr.equals(replayed_expr),
    'original_manuscript_modules_classifier_source_predictor_unchanged': True,
    'both_preparation_and_scoring_use_final_prescoring_protocol': True,
    'protocol_amendment_note': 'The fair 20-subject target-mean comparator was added before first prediction scoring. Initial preparation recorded the earlier metadata-only protocol hash; the replay preserves that manifest and records the final hash in both steps.',
    'bootstrap_intervals_condition_on_fixed_source_model_and_20_calibration_subjects': True,
    'age_stratum_analysis_was_post_protocol_exploratory_diagnostic': True,
    'longitudinal_visits_overlap_primary_evaluation_and_are_not_a_third_cohort': True,
    'clipping_counts': clipping,
    'missing_predictor_genes': coverage.loc[~coverage.available, 'gene'].to_list(),
}
(OUT / 'verification_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
