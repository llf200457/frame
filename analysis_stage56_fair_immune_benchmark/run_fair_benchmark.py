"""Finite, equal-budget external benchmark; no model/seed/feature search."""
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
threadpool_limits(limits=1)
BASE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else BASE
OUT.mkdir(parents=True, exist_ok=True)
DATA = BASE.parent
S = DATA / 'analysis_stage53_measured_blood_anchor'
E = DATA / 'analysis_stage54_soundlife_external'
P = json.loads((BASE / 'execution_protocol.json').read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(y, p):
    d = p - y
    denominator = np.sum((y - y.mean()) ** 2)
    return dict(R2=float(1 - np.sum(d * d) / denominator) if denominator > 0 else np.nan,
                MAE=float(np.mean(np.abs(d))), RMSE=float(np.sqrt(np.mean(d * d))),
                Spearman_r=float(spearmanr(y, p).statistic) if np.ptp(p) > 0 and np.ptp(y) > 0 else np.nan,
                bias=float(d.mean()))


def boot_intervals(y, p, draws):
    yy, pp = y[draws], p[draws]
    dd = pp - yy
    denom = np.sum((yy - yy.mean(axis=1, keepdims=True)) ** 2, axis=1)
    values = dict(R2=1 - np.sum(dd * dd, axis=1) / denom, MAE=np.abs(dd).mean(axis=1),
                  RMSE=np.sqrt((dd * dd).mean(axis=1)), bias=dd.mean(axis=1))
    result = {}
    for key, arr in values.items():
        result[key + '_CI_low'], result[key + '_CI_high'] = map(float, np.quantile(arr, [.025, .975]))
    return result


preserved = {n: sha(DATA / n) for n in P['preserve_files']}
assert preserved == P['preserve_files']
frozen = json.loads((BASE / 'source_baseline_parameters.json').read_text(encoding='utf-8'))
prep = json.loads((BASE / 'preparation_manifest.json').read_text(encoding='utf-8'))
assert sha(BASE / 'source_baseline_parameters.json') == prep['source_model_sha256']
primary = pd.read_csv(E / 'primary_one_visit_metadata.csv')
names = primary['sample'].tolist()
assert len(primary) == 94 and primary.subject.is_unique and primary['sample'].is_unique
calmask = primary.role.eq('calibration_subject').to_numpy()
heldmask = primary.role.eq('heldout_subject').to_numpy()
assert calmask.sum() == 20 and heldmask.sum() == 74
assert set(primary.subject[calmask]).isdisjoint(set(primary.subject[heldmask]))
fixed_subjects = pd.read_csv(E / 'fixed_calibration_subjects.csv').subject.tolist()
assert set(fixed_subjects) == set(primary.subject[calmask])
source_cov = primary[['age', 'male', 'sex_missing']].to_numpy(float)
marker_scores = pd.read_csv(BASE / 'target_marker_scores.csv', index_col='sample')
marker_x = np.column_stack([source_cov, marker_scores.loc[names, list(frozen['marker_gene_sets'])].to_numpy(float)])
predictions, raw_predictions = {}, {}
for name, x in [('age_sex', source_cov), ('marker7', marker_x)]:
    model = frozen['models'][name]
    raw_predictions[name] = ((x - np.array(model['mean'])) / np.array(model['scale'])) @ np.array(model['coef']) + model['intercept']
    predictions[name] = np.clip(raw_predictions[name], 0, 100)

locked = np.load(S / 'locked_neutrophil_factor_predictor.npz', allow_pickle=False)
expr = pd.read_pickle(E / 'SoundLife_model_gene_TPM.pkl', compression='gzip')
genes = locked['genes'].tolist()
position = {g: i for i, g in enumerate(genes)}
available = [g for g in genes if g in expr.index]
index = np.array([position[g] for g in available])
z = np.zeros((94, len(genes)))
z[:, index] = (np.log2(expr.loc[available, names].to_numpy(float).T + 1) - locked['gene_mean'][index]) / locked['gene_sd'][index]
modules = z @ locked['module_weights']
x = np.column_stack([source_cov, modules])
raw_predictions['locked8'] = ((x - locked['feature_mean']) / locked['feature_scale']) @ locked['coefficients'] + locked['intercept']
predictions['locked8'] = np.clip(raw_predictions['locked8'], 0, 100)
old = pd.read_csv(E / 'primary_one_visit_predictions.csv').set_index('sample').loc[names]
assert np.allclose(old.locked8.to_numpy(), predictions['locked8'], atol=1e-10, rtol=0)
assert np.allclose(old.source_age_sex.to_numpy(), predictions['age_sex'], atol=1e-10, rtol=0)

y = primary.target_neutrophils_percent.to_numpy(float)
assert np.all(np.isfinite(y)) and np.all((y >= 0) & (y <= 100))
frame = primary[['sample', 'subject', 'role', 'age', 'male', 'sex_missing']].copy()
frame['observed_neutrophils_percent'] = y
cal_parameters = {}
clipping = []
for name in P['models']:
    p = predictions[name]
    delta = float(np.mean(y[calmask] - p[calmask]))
    affine = LinearRegression().fit(p[calmask, None], y[calmask])
    cal_parameters[name] = dict(n=20, intercept_delta=delta, affine_intercept=float(affine.intercept_),
                                affine_slope=float(affine.coef_[0]))
    for regime, values in [('unadapted', raw_predictions[name]), ('intercept', p + delta),
                            ('affine', affine.predict(p[:, None]))]:
        clipped = np.clip(values, 0, 100)
        frame[regime + '__' + name] = clipped
        clipping.append(dict(model=name, regime=regime, preclip_below0_all94=int((values < 0).sum()),
                             preclip_above100_all94=int((values > 100).sum()),
                             preclip_outside_heldout74=int(((values[heldmask] < 0) | (values[heldmask] > 100)).sum())))
frame['calibration_mean'] = float(y[calmask].mean())
frame.to_csv(OUT / 'primary_predictions.csv', index=False)
held = frame.loc[heldmask].reset_index(drop=True)
held.to_csv(OUT / 'heldout_predictions.csv', index=False)
frame.loc[calmask, ['sample', 'subject', 'role']].to_csv(OUT / 'calibration_subject_roles.csv', index=False)
held[['sample', 'subject', 'role']].to_csv(OUT / 'evaluation_subject_roles.csv', index=False)
pd.DataFrame(clipping).to_csv(OUT / 'clipping_audit.csv', index=False)
(OUT / 'calibration_parameters.json').write_text(json.dumps(cal_parameters, indent=2), encoding='utf-8')
old_cal = json.loads((E / 'calibration_parameters.json').read_text(encoding='utf-8'))
for key, oldkey in [('intercept_delta', 'intercept_delta'), ('affine_intercept', 'affine_intercept'), ('affine_slope', 'affine_slope')]:
    assert abs(cal_parameters['locked8'][key] - old_cal[oldkey]) < 1e-10

rng = np.random.default_rng(P['bootstrap']['seed'])
draws = rng.integers(0, len(held), size=(P['bootstrap']['draws'], len(held)))
np.save(OUT / 'shared_bootstrap_draws.npy', draws)
yy = held.observed_neutrophils_percent.to_numpy(float)
rows = []
for regime in P['calibration_regimes']:
    for model in P['models']:
        p = held[regime + '__' + model].to_numpy(float)
        row = dict(scope='matched_heldout74', regime=regime, model=model, n=74,
                   **metrics(yy, p), **boot_intervals(yy, p, draws))
        rows.append(row)
        print(regime, model, metrics(yy, p), flush=True)
pmean = held.calibration_mean.to_numpy(float)
rows.append(dict(scope='matched_heldout74', regime='constant', model='calibration_mean', n=74,
                 **metrics(yy, pmean), **boot_intervals(yy, pmean, draws)))
pd.DataFrame(rows).to_csv(OUT / 'external_performance.csv', index=False)
pd.DataFrame([dict(scope='descriptive_all94', regime='unadapted', model=model, n=94,
                   **metrics(y, frame['unadapted__' + model].to_numpy(float)))
              for model in P['models']]).to_csv(OUT / 'all94_unadapted_diagnostics.csv', index=False)

contrasts = []
for regime in P['calibration_regimes']:
    for extended, baseline in combinations(P['models'], 2):
        d = np.abs(yy - held[regime + '__' + baseline].to_numpy(float)) - np.abs(yy - held[regime + '__' + extended].to_numpy(float))
        lo, hi = np.quantile(d[draws].mean(axis=1), [.025, .975])
        primary_comparison = regime == 'intercept' and extended == 'locked8' and baseline == 'marker7'
        contrasts.append(dict(regime=regime, baseline=baseline, extended=extended, n=74,
                              MAE_gain=float(d.mean()), CI_low=float(lo), CI_high=float(hi),
                              primary_comparison=primary_comparison,
                              interpretation='primary' if primary_comparison else 'descriptive_not_multiplicity_corrected'))
pd.DataFrame(contrasts).to_csv(OUT / 'paired_MAE_comparisons.csv', index=False)
mean_contrasts = []
for regime in ['intercept', 'affine']:
    for model in P['models']:
        d = np.abs(yy - pmean) - np.abs(yy - held[regime + '__' + model].to_numpy(float))
        lo, hi = np.quantile(d[draws].mean(axis=1), [.025, .975])
        mean_contrasts.append(dict(regime=regime, model=model, n=74, MAE_gain_over_local_mean=float(d.mean()),
                                   CI_low=float(lo), CI_high=float(hi), interpretation='descriptive'))
pd.DataFrame(mean_contrasts).to_csv(OUT / 'local_mean_comparisons.csv', index=False)
strata = []
for label, mask in [('age_below45', held.age.lt(45).to_numpy()), ('age_at_least45', held.age.ge(45).to_numpy())]:
    for model in P['models']:
        strata.append(dict(stratum=label, model=model, regime='intercept', n=int(mask.sum()),
                           **metrics(yy[mask], held['intercept__' + model].to_numpy(float)[mask])))
pd.DataFrame(strata).to_csv(OUT / 'age_stratum_diagnostics.csv', index=False)
primary_result = next(r for r in contrasts if r['primary_comparison'])
manifest = dict(status='Completed finite matched-budget retrospective benchmark', source_n=125,
                calibration_n=20, independent_evaluation_subject_n=74, new_untouched_cohort=False,
                same_roles_as_stage54=True, baseline_marker_gene_n=61, baseline_marker_score_n=7,
                primary_comparison=primary_result, primary_superiority_gate_passed=bool(primary_result['CI_low'] > 0),
                stop_rule_applies=bool(primary_result['MAE_gain'] <= 0),
                original_source_model_unchanged=True, AD_triage_benefit_established=False,
                general_algorithm_superiority_established=False,
                protocol_sha256=sha(BASE / 'execution_protocol.json'), baseline_parameters_sha256=sha(BASE / 'source_baseline_parameters.json'),
                bootstrap_sha256=sha(OUT / 'shared_bootstrap_draws.npy'),
                stage54_consistency=dict(locked8_prediction_max_difference=float(np.max(np.abs(old.locked8.to_numpy() - predictions['locked8']))),
                                         age_sex_prediction_max_difference=float(np.max(np.abs(old.source_age_sex.to_numpy() - predictions['age_sex'])))),
                preserved_before=preserved, preserved_after={n: sha(DATA / n) for n in preserved})
assert manifest['preserved_before'] == manifest['preserved_after']
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('PRIMARY', primary_result, 'SUPERIORITY_GATE', manifest['primary_superiority_gate_passed'], flush=True)
