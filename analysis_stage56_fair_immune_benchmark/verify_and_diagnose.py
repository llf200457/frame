"""Independent formulas, deterministic replay, and explicitly post-primary diagnostics."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
threadpool_limits(limits=1)
OUT = Path(__file__).resolve().parent
DATA = OUT.parent
S = DATA / 'analysis_stage53_measured_blood_anchor'
E = DATA / 'analysis_stage54_soundlife_external'
P = json.loads((OUT / 'execution_protocol.json').read_text(encoding='utf-8'))
D = json.loads((OUT / 'post_primary_diagnostic_protocol.json').read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


frozen = json.loads((OUT / 'source_baseline_parameters.json').read_text(encoding='utf-8'))
params = json.loads((OUT / 'calibration_parameters.json').read_text(encoding='utf-8'))
pred = pd.read_csv(OUT / 'primary_predictions.csv')
cal = pred[pred.role.eq('calibration_subject')].copy()
held = pred[pred.role.eq('heldout_subject')].copy().reset_index(drop=True)
assert len(cal) == 20 and len(held) == 74 and set(cal.subject).isdisjoint(set(held.subject))
yy = held.observed_neutrophils_percent.to_numpy(float)
checks = []
performance = pd.read_csv(OUT / 'external_performance.csv')
for row in performance.itertuples():
    p = held.calibration_mean.to_numpy(float) if row.model == 'calibration_mean' else held[row.regime + '__' + row.model].to_numpy(float)
    independently = dict(R2=r2_score(yy, p), MAE=mean_absolute_error(yy, p),
                         RMSE=np.sqrt(mean_squared_error(yy, p)), bias=(np.sum(p) - np.sum(yy)) / len(yy),
                         Spearman_r=pd.Series(yy).rank().corr(pd.Series(p).rank()) if np.ptp(p) else np.nan)
    for key, value in independently.items():
        old = getattr(row, key)
        if np.isnan(value) and np.isnan(old):
            continue
        diff = abs(float(old) - float(value))
        checks.append(dict(check='independent_' + key, model=row.model, regime=row.regime, max_abs_difference=diff))
        assert diff < 1e-10
for name in P['models']:
    p = cal['unadapted__' + name].to_numpy(float)
    y = cal.observed_neutrophils_percent.to_numpy(float)
    delta = float(np.mean(y) - np.mean(p))
    slope = float(np.sum((p - p.mean()) * (y - y.mean())) / np.sum((p - p.mean()) ** 2))
    intercept = float(y.mean() - slope * p.mean())
    for key, value in [('intercept_delta', delta), ('affine_slope', slope), ('affine_intercept', intercept)]:
        diff = abs(params[name][key] - value)
        checks.append(dict(check='independent_' + key, model=name, regime='calibration', max_abs_difference=diff))
        assert diff < 1e-10

# Recompute five complete raw-count predictions without using saved target TPM/features.
meta = pd.read_csv(E / 'primary_one_visit_metadata.csv').set_index('sample')
ann = pd.read_csv(E / 'gene_length_annotation.csv')
lengths = ann.exon_union_length_bp.to_numpy(float)
varpos = {g: i for i, g in enumerate(ann['name'])}
locked = np.load(S / 'locked_neutrophil_factor_predictor.npz', allow_pickle=False)
modelgenes = locked['genes'].tolist()
available = [g for g in modelgenes if g in varpos]
mi = np.array([modelgenes.index(g) for g in available])
ri = np.array([varpos[g] for g in available])
raw_checks = []
with h5py.File(E / 'raw/sound-life_whole-blood_no-stim.h5ad', 'r') as h:
    for rownum in [0, 19, 39, 59, 93]:
        row = pred.iloc[rownum]
        r = meta.loc[row['sample']]
        rate = h['X'][int(r.h5_row), :].astype(float) / lengths
        tpm = rate * 1e6 / rate.sum()
        clinical = np.array([r.age, r.male, r.sex_missing], dtype=float)
        marker = np.array([np.log2(tpm[[varpos[g] for g in gs]] + 1).mean() for gs in frozen['marker_gene_sets'].values()])
        gene_z = np.zeros(len(modelgenes))
        gene_z[mi] = (np.log2(tpm[ri] + 1) - locked['gene_mean'][mi]) / locked['gene_sd'][mi]
        features = np.r_[clinical, gene_z @ locked['module_weights']]
        independent = {'locked8': float(np.sum((features - locked['feature_mean']) / locked['feature_scale'] * locked['coefficients']) + locked['intercept'])}
        for name, x in [('age_sex', clinical), ('marker7', np.r_[clinical, marker])]:
            m = frozen['models'][name]
            independent[name] = float(np.sum((x - np.array(m['mean'])) / np.array(m['scale']) * np.array(m['coef'])) + m['intercept'])
        for name, value in independent.items():
            p = np.clip(value, 0, 100)
            c = params[name]
            values = dict(unadapted=p, intercept=np.clip(p + c['intercept_delta'], 0, 100),
                          affine=np.clip(c['affine_intercept'] + c['affine_slope'] * p, 0, 100))
            for regime, pp in values.items():
                diff = abs(row[regime + '__' + name] - pp)
                raw_checks.append(dict(sample=row['sample'], model=name, regime=regime, abs_difference=diff))
                assert diff < 1e-10
pd.DataFrame(raw_checks).to_csv(OUT / 'independent_raw_prediction_checks.csv', index=False)
pd.DataFrame(checks).to_csv(OUT / 'independent_formula_checks.csv', index=False)

# Exact same seed, source artifacts and inputs; replay in its own subdirectory.
replay = OUT / 'replay'
with (OUT / 'benchmark_replay_execution.log').open('w', encoding='utf-8') as f:
    subprocess.run([sys.executable, str(OUT / 'run_fair_benchmark.py'), str(replay)], stdout=f, stderr=subprocess.STDOUT, check=True)
replay_checks = []
for path in sorted(OUT.glob('*.csv')):
    other = replay / path.name
    if not other.exists():
        continue
    a, b = sha(path), sha(other)
    replay_checks.append(dict(file=path.name, sha256=a, replay_sha256=b, byte_identical=a == b))
    assert a == b
for name in ['shared_bootstrap_draws.npy', 'calibration_parameters.json']:
    assert sha(OUT / name) == sha(replay / name)
pd.DataFrame(replay_checks).to_csv(OUT / 'full_replay_checks.csv', index=False)

# Influence only: every subject retained. Positive difference favors original8.
d = np.abs(yy - held.intercept__marker7.to_numpy(float)) - np.abs(yy - held.intercept__locked8.to_numpy(float))
influence = held[['sample', 'subject', 'age', 'observed_neutrophils_percent']].copy()
influence['absolute_error_locked8'] = np.abs(yy - held.intercept__locked8.to_numpy(float))
influence['absolute_error_marker7'] = np.abs(yy - held.intercept__marker7.to_numpy(float))
influence['paired_error_gain'] = d
influence['leave_one_subject_out_mean_gain'] = (d.sum() - d) / (len(d) - 1)
influence.to_csv(OUT / 'all_subject_influence_diagnostics.csv', index=False)
influence_summary = dict(n=74, mean_gain=float(d.mean()), median_subject_gain=float(np.median(d)),
                         subjects_with_lower_error_locked8=int((d > 0).sum()), subjects_with_equal_error=int((d == 0).sum()),
                         subjects_with_lower_error_marker7=int((d < 0).sum()),
                         leave_one_out_mean_gain_min=float(influence.leave_one_subject_out_mean_gain.min()),
                         leave_one_out_mean_gain_max=float(influence.leave_one_subject_out_mean_gain.max()),
                         exclusions=0, role='post-primary descriptive diagnostic')

# Post-primary two-sample bootstrap, NOT a new model/group search.
rng = np.random.default_rng(20262059)
caldraws = rng.integers(0, 20, size=(2000, 20))
helddraws = rng.integers(0, 74, size=(2000, 74))
np.savez(OUT / 'calibration_evaluation_bootstrap_draws.npz', calibration=caldraws, evaluation=helddraws)
cal_y = cal.observed_neutrophils_percent.to_numpy(float)
ystar = yy[helddraws]
loss, allmetrics = {}, {}
for name in P['models']:
    cp = cal['unadapted__' + name].to_numpy(float)
    hp = held['unadapted__' + name].to_numpy(float)
    delta = np.mean(cal_y[caldraws] - cp[caldraws], axis=1)
    pstar = np.clip(hp[helddraws] + delta[:, None], 0, 100)
    err = pstar - ystar
    loss[name] = np.abs(err).mean(axis=1)
    allmetrics[name] = dict(MAE=loss[name], R2=1 - (err * err).sum(axis=1) / ((ystar - ystar.mean(axis=1, keepdims=True)) ** 2).sum(axis=1))
rows = []
for name, values in allmetrics.items():
    for metric, arr in values.items():
        lo, hi = np.quantile(arr, [.025, .975])
        point = performance.loc[(performance.model == name) & (performance.regime == 'intercept'), metric].iloc[0]
        rows.append(dict(type='metric', model=name, baseline='', metric=metric, point_estimate=float(point),
                         CI_low=float(lo), CI_high=float(hi), bootstrap_unit='independent calibration20 and evaluation74', post_primary=True))
for baseline in ['marker7', 'age_sex']:
    values = loss[baseline] - loss['locked8']
    lo, hi = np.quantile(values, [.025, .975])
    original = float((np.abs(yy - held['intercept__' + baseline].to_numpy()) - np.abs(yy - held.intercept__locked8.to_numpy())).mean())
    rows.append(dict(type='paired_gain', model='locked8', baseline=baseline, metric='MAE_gain',
                     point_estimate=original, CI_low=float(lo), CI_high=float(hi),
                     bootstrap_unit='independent calibration20 and evaluation74', post_primary=True))
pd.DataFrame(rows).to_csv(OUT / 'calibration_uncertainty_diagnostics.csv', index=False)
manifest = dict(verified=True, numeric_formula_check_n=len(checks), raw_prediction_check_n=len(raw_checks),
                max_formula_difference=max(v['max_abs_difference'] for v in checks),
                max_raw_prediction_difference=max(v['abs_difference'] for v in raw_checks),
                replay_CSV_byte_identical_n=len(replay_checks), replay_all_CSV_identical=True,
                bootstrap_draws_replay_identical=True, influence=influence_summary,
                post_primary_diagnostic_protocol_sha256=sha(OUT / 'post_primary_diagnostic_protocol.json'),
                calibration_uncertainty=rows, preserved_after={n: sha(DATA / n) for n in P['preserve_files']})
assert manifest['preserved_after'] == P['preserve_files']
(OUT / 'verification_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Verification completed:', len(replay_checks), 'identical CSVs;', len(raw_checks), 'raw predictions;', len(checks), 'formulas', flush=True)
print('Influence:', influence_summary, flush=True)
print('Calibration uncertainty paired gains:', [r for r in rows if r['type'] == 'paired_gain'], flush=True)
