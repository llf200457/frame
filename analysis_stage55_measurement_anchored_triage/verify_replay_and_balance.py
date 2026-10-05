"""Independent arithmetic, one full replay and posthoc composition diagnostic."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
threadpool_limits(limits=1)
OUT = Path(__file__).resolve().parent
DATA = OUT.parent
SEED = 20261003


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


names = [
    'fixed_module_directions.csv', 'RNA_measured_factor_projection_records.csv',
    'RNA_measured_factor_associations.csv', 'all330_source_support_reference.csv',
    'all_AD_array_audit_records.csv', 'support_selection_performance.csv',
    'fixed_source_gate_performance.csv', 'drift_partition_summary.csv',
    'paired_retained_Brier_gains.csv', 'immune_direction_error_associations.csv',
    'independent_batch_genotype_audit.csv',
]
before = {n: sha(OUT / n) for n in names}
manifest = json.loads((OUT / 'run_manifest.json').read_text(encoding='utf-8'))
par = json.loads((DATA / 'metadata/stage22/primary_classifier_parameters.json').read_text(encoding='utf-8'))
model = np.load(DATA / 'analysis_stage53_measured_blood_anchor/locked_neutrophil_factor_predictor.npz')
reference = np.load(OUT / 'anchored_support_reference.npz')
saved_reference = {k: reference[k].copy() for k in reference.files}
direction = pd.read_csv(OUT / 'fixed_module_directions.csv')
assert np.allclose(direction.original_AD_raw_module_coefficient,
                   np.asarray(par['coefficients'][0]) / np.asarray(par['scaler_scale']), atol=1e-14)
assert np.allclose(direction.measured_neutrophil_raw_module_direction,
                   model['coefficients'][3:] / model['feature_scale'][3:], atol=1e-14)

# Independent eigenspace evaluation of Mahalanobis projection and residual.
eigval, eigvec = np.linalg.eigh(reference['covariance'])
assert eigval.min() > 0
records = pd.read_csv(OUT / 'all_AD_array_audit_records.csv')
decomposition_errors = []
for cohort, frame in records.groupby('cohort', sort=True):
    m = pd.read_csv(DATA / f'analysis_stage51_cell_conditioned_transport/{cohort}_frozen_module_input.csv', index_col=0).loc[frame['sample'], par['features']].to_numpy()
    delta = m - reference['mean']
    whitened = delta @ eigvec / np.sqrt(eigval)
    axis = np.sqrt(eigval) * (eigvec.T @ reference['measured_direction'])
    axis /= np.linalg.norm(axis)
    signed = whitened @ axis
    orth = whitened - signed[:, None] * axis[None, :]
    errors = [np.max(np.abs(np.sum(whitened ** 2, axis=1) - frame.original_module_drift)),
              np.max(np.abs(signed - frame.signed_immune_direction_index)),
              np.max(np.abs(np.sum(orth ** 2, axis=1) - frame.measurement_anchored_residual_distance))]
    assert max(errors) < 1e-8
    decomposition_errors.extend(float(v) for v in errors)
    y = frame.status.eq('AD').to_numpy(int)
    assert np.max(np.abs((frame.p_frozen - y) ** 2 - frame.squared_error)) < 1e-12

# Independent recomputation of signed RNA projections from previously audited
# TPM input and source scalers; target eight missing genes are source-z zero.
proj = pd.read_csv(OUT / 'RNA_measured_factor_projection_records.csv')
target = proj[proj.cohort.eq('SoundLife94')]
expr = pd.read_pickle(DATA / 'analysis_stage54_soundlife_external/SoundLife_model_gene_TPM.pkl', compression='gzip')
genes = model['genes'].tolist()
idx = np.array([i for i, g in enumerate(genes) if g in expr.index])
g = [genes[i] for i in idx]
samples = target['sample'].to_list()
z = np.zeros((len(samples), len(genes)))
z[:, idx] = (np.log2(expr.loc[g, samples].to_numpy().T + 1) - model['gene_mean'][idx]) / model['gene_sd'][idx]
module = np.column_stack([np.sum(z * model['module_weights'][:, j], axis=1) for j in range(8)])
relative = module - model['feature_mean'][3:]
b_ad = direction.original_AD_raw_module_coefficient.to_numpy()
b_neut = direction.measured_neutrophil_raw_module_direction.to_numpy()
projection_errors = [np.max(np.abs(relative @ b_ad - target.AD_weighted_RNA_projection)),
                     np.max(np.abs(relative @ b_neut - target.neutrophil_module_only_projection))]
assert max(projection_errors) < 1e-9

checks = []
for row in pd.read_csv(OUT / 'RNA_measured_factor_associations.csv').to_dict('records'):
    f = proj[proj.cohort.eq(row['cohort'])]
    columns = [np.ones(len(f)), rankdata(f.age), f.male.to_numpy(), f.sex_missing.to_numpy()]
    if row['cohort'] == 'Overmyer125':
        columns.append(f.COVID.to_numpy())
    cov = np.column_stack(columns)
    rx = rankdata(f[row['projection']]); ry = rankdata(f.measured_neutrophils_percent)
    ax = rx - cov @ (np.linalg.pinv(cov) @ rx)
    ay = ry - cov @ (np.linalg.pinv(cov) @ ry)
    value = float(np.dot(ax, ay) / np.sqrt(np.dot(ax, ax) * np.dot(ay, ay)))
    error = abs(value - row['partial_rank_r'])
    assert error < 1e-12
    checks.append(dict(cohort=row['cohort'], projection=row['projection'], partial_rank_absolute_error=error))
pd.DataFrame(checks).to_csv(OUT / 'independent_association_checks.csv', index=False)

print('Independent decomposition, factor-projection and association checks passed.', flush=True)
with (OUT / 'full_replay_execution.log').open('w', encoding='utf-8') as stream:
    result = subprocess.run([sys.executable, '-u', str(OUT / 'run_anchored_triage_audit.py')],
                            cwd=OUT, stdout=stream, stderr=subprocess.STDOUT)
assert result.returncode == 0
after = {n: sha(OUT / n) for n in names}
pd.DataFrame([dict(file=n, before_sha256=before[n], after_sha256=after[n],
                   byte_identical=before[n] == after[n]) for n in names]).to_csv(OUT / 'full_replay_checks.csv', index=False)
assert before == after
replay = np.load(OUT / 'anchored_support_reference.npz')
assert all(np.array_equal(v, replay[k]) for k, v in saved_reference.items())
for name, digest in manifest['preserved_before'].items():
    assert sha(DATA / name) == digest

# Composition diagnostic specified separately after primary score inspection.
scores = ['original_module_drift', 'frozen_prediction_uncertainty',
          'measurement_anchored_immune_distance', 'measurement_anchored_residual_distance']
balance_rows, gain_rows = [], []


def select(score, coverage):
    return np.argsort(score, kind='stable')[:int(np.floor(len(score) * coverage))]


def balanced(y, loss, take):
    means = [loss[take][y[take] == k].mean() if np.any(y[take] == k) else np.nan for k in [0, 1]]
    return float(np.mean(means))


for cohort, frame in records.groupby('cohort', sort=True):
    frame = frame.sort_values('sample').reset_index(drop=True)
    y = frame.status.eq('AD').to_numpy(int)
    loss = frame.squared_error.to_numpy(float)
    for coverage in [.8, .6]:
        for score in scores:
            take = select(frame[score].to_numpy(), coverage)
            for status, label in [('AD', 1), ('CTL', 0)]:
                cls = take[y[take] == label]
                balance_rows.append(dict(cohort=cohort, score=score, coverage=coverage, status=status,
                                         class_n=int(np.sum(y == label)), class_retained_n=len(cls),
                                         class_retained_coverage=len(cls) / np.sum(y == label),
                                         class_retained_brier=float(loss[cls].mean()),
                                         balanced_retained_brier=balanced(y, loss, take)))
    if cohort == 'GSE63060':
        continue
    classes = [np.flatnonzero(y == k) for k in [0, 1]]
    rng = np.random.default_rng(SEED + 455)
    draws = [np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in classes]) for _ in range(2000)]
    candidate = frame.measurement_anchored_residual_distance.to_numpy()
    for baseline in ['original_module_drift', 'frozen_prediction_uncertainty']:
        score = frame[baseline].to_numpy()
        gain = balanced(y, loss, select(score, .8)) - balanced(y, loss, select(candidate, .8))
        boot = [balanced(y[i], loss[i], select(score[i], .8)) - balanced(y[i], loss[i], select(candidate[i], .8)) for i in draws]
        lo, hi = np.quantile(boot, [.025, .975])
        gain_rows.append(dict(cohort=cohort, baseline=baseline, balanced_Brier_gain=gain,
                              CI_low=lo, CI_high=hi, post_primary_robustness=True,
                              primary_success_gate_not_changed=True))
pd.DataFrame(balance_rows).to_csv(OUT / 'class_composition_diagnostics.csv', index=False)
pd.DataFrame(gain_rows).to_csv(OUT / 'balanced_Brier_gain_diagnostics.csv', index=False)
verification = dict(replayed_main_pipeline_CSVs=len(names), all_main_CSVs_byte_identical=True,
                    independent_geometry_max_error=max(decomposition_errors),
                    independent_target_RNA_projection_max_error=float(max(projection_errors)),
                    independent_rank_association_max_error=max(r['partial_rank_absolute_error'] for r in checks),
                    all_original_files_and_locked_models_unchanged=True,
                    class_balanced_diagnostic_was_post_primary_and_did_not_change_gate=True,
                    source_reference_NPZ_arrays_identical_on_replay=True,
                    all55_cohorts_previously_inspected_and_exploratory=True)
(OUT / 'verification_manifest.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(json.dumps(verification, indent=2), flush=True)
print(pd.DataFrame(gain_rows).to_string(index=False), flush=True)
