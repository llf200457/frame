"""One official immune-score ridge baseline; fixed data, alpha and calibration roles."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
threadpool_limits(limits=1)
BASE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else BASE
OUT.mkdir(parents=True, exist_ok=True)
DATA = BASE.parent
S = DATA / 'analysis_stage53_measured_blood_anchor'
B = DATA / 'analysis_stage56_fair_immune_benchmark'
A = DATA / 'analysis_stage57_external_resource_audit'
P = json.loads((BASE / 'execution_protocol.json').read_text(encoding='utf-8'))
IMMUNE = ['T cells', 'CD8 T cells', 'Cytotoxic lymphocytes', 'NK cells', 'B lineage',
          'Monocytic lineage', 'Myeloid dendritic cells', 'Neutrophils']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, p):
    d = p - y
    return dict(R2=float(1 - (d * d).sum() / ((y - y.mean()) ** 2).sum()),
                MAE=float(np.abs(d).mean()), RMSE=float(np.sqrt((d * d).mean())),
                Spearman_r=float(spearmanr(y, p).statistic), bias=float(d.mean()))


preserved = {n: sha(DATA / n) for n in P['preserve_files']}
assert preserved == P['preserve_files']
assert sha(A / 'raw/MCPcounter.R') == json.loads((A / 'resource_audit_manifest.json').read_text())['mcp_audit']['code_sha256']
checks = []
sets = pd.read_csv(A / 'raw/MCPcounter_genes.txt', sep='\t')
scores = {}
for scope in ['source', 'target']:
    xp = pd.read_csv(BASE / (scope + '_official_marker_logTPM.csv'), index_col=0)
    native = pd.read_csv(BASE / (scope + '_official_MCPcounter_scores.csv'), index_col=0)
    assert len(native) == 10 and list(xp.columns) == list(native.columns)
    for cell in native.index:
        genes = [g for g in sets.loc[sets['Cell population'].eq(cell), 'HUGO symbols'] if g in xp.index]
        difference = float(np.max(np.abs(xp.loc[genes].mean(axis=0).to_numpy() - native.loc[cell].to_numpy())))
        assert difference < 1e-12
        checks.append(dict(check='official_R_vs_independent_mean', scope=scope, cell=cell, max_difference=difference, available_gene_n=len(genes)))
    scores[scope] = native
source = pd.read_csv(S / 'paired_clinical_metadata.csv')
sx = np.column_stack([source[['age', 'male', 'sex_missing']].to_numpy(float), scores['source'].loc[IMMUNE, source['sample']].T.to_numpy(float)])
sy = source.target_neutrophils_percent.to_numpy(float)
scaler = StandardScaler().fit(sx)
zs = scaler.transform(sx)
model = Ridge(alpha=10.0).fit(zs, sy)
coef_formula = np.linalg.solve(zs.T @ zs + 10 * np.eye(zs.shape[1]), zs.T @ (sy - sy.mean()))
fitdiff = float(np.max(np.abs(model.coef_ - coef_formula)))
assert fitdiff < 1e-10
checks.append(dict(check='source_ridge_closed_form', scope='source', cell='all8', max_difference=fitdiff, available_gene_n=66))
parameters = dict(source_n=125, alpha=10.0, feature_names=['age', 'male', 'sex_missing'] + IMMUNE,
                  mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=model.coef_.tolist(), intercept=float(model.intercept_),
                  upstream_commit='b6eac73e91c246fcff0bb1a5c68a816cd588fc48', official_R_run=True,
                  common_gene_support=True, non_immune_scores_excluded_before_fit=True,
                  protocol_sha256=sha(BASE / 'execution_protocol.json'))
(OUT / 'source_official_MCP8_parameters.json').write_text(json.dumps(parameters, indent=2), encoding='utf-8')
print('Source official_MCP8 parameters frozen', sha(OUT / 'source_official_MCP8_parameters.json'), flush=True)

# Reuse prior subject ordering and locked predictions unchanged.
frame = pd.read_csv(B / 'primary_predictions.csv')
meta = pd.read_csv(DATA / 'analysis_stage54_soundlife_external/primary_one_visit_metadata.csv').set_index('sample').loc[frame['sample']]
cov = meta[['age', 'male', 'sex_missing']].to_numpy(float)
tx = np.column_stack([cov, scores['target'].loc[IMMUNE, frame['sample']].T.to_numpy(float)])
raw = model.predict(scaler.transform(tx))
p = np.clip(raw, 0, 100)
calmask = frame.role.eq('calibration_subject').to_numpy()
heldmask = frame.role.eq('heldout_subject').to_numpy()
assert calmask.sum() == 20 and heldmask.sum() == 74
assert set(frame.subject[calmask]).isdisjoint(set(frame.subject[heldmask]))
y = frame.observed_neutrophils_percent.to_numpy(float)
delta = float((y[calmask] - p[calmask]).mean())
affine = LinearRegression().fit(p[calmask, None], y[calmask])
frame['unadapted__official_MCP8'] = p
frame['intercept__official_MCP8'] = np.clip(p + delta, 0, 100)
frame['affine__official_MCP8'] = np.clip(affine.predict(p[:, None]), 0, 100)
frame.to_csv(OUT / 'primary_predictions.csv', index=False)
held = frame[heldmask].reset_index(drop=True)
held.to_csv(OUT / 'heldout_predictions.csv', index=False)
calibration = dict(calibration_subject_n=20, evaluation_subject_n=74, offset=delta,
                   affine_intercept=float(affine.intercept_), affine_slope=float(affine.coef_[0]),
                   preclip_source_outside_all94=int(((raw < 0) | (raw > 100)).sum()),
                   preclip_offset_outside_all94=int(((p + delta < 0) | (p + delta > 100)).sum()))
(OUT / 'calibration_parameters.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')
yy = held.observed_neutrophils_percent.to_numpy(float)
draws = np.load(B / 'shared_bootstrap_draws.npy')
rows = []
for regime in ['unadapted', 'intercept', 'affine']:
    for name in ['locked8', 'marker7', 'official_MCP8']:
        pp = held[regime + '__' + name].to_numpy(float)
        row = dict(model=name, regime=regime, n=74, **metrics(yy, pp))
        ds = pp[draws] - yy[draws]
        for key, vals in [('MAE', np.abs(ds).mean(axis=1)), ('R2', 1 - (ds * ds).sum(axis=1) / ((yy[draws] - yy[draws].mean(axis=1, keepdims=True)) ** 2).sum(axis=1))]:
            row[key + '_CI_low'], row[key + '_CI_high'] = map(float, np.quantile(vals, [.025, .975]))
        rows.append(row)
        assert abs(row['R2'] - r2_score(yy, pp)) < 1e-10 and abs(row['MAE'] - mean_absolute_error(yy, pp)) < 1e-10
        print(regime, name, metrics(yy, pp), flush=True)
pd.DataFrame(rows).to_csv(OUT / 'external_performance.csv', index=False)
contrasts = []
for regime in ['unadapted', 'intercept', 'affine']:
    d = np.abs(yy - held[regime + '__official_MCP8'].to_numpy()) - np.abs(yy - held[regime + '__locked8'].to_numpy())
    lo, hi = np.quantile(d[draws].mean(axis=1), [.025, .975])
    contrasts.append(dict(regime=regime, baseline='official_MCP8', extended='locked8', n=74,
                          MAE_gain=float(d.mean()), CI_low=float(lo), CI_high=float(hi), uncertainty='fixed20_calibration'))

# Primary uncertainty uses identical saved two-group draws, not a new seed search.
bd = np.load(B / 'calibration_evaluation_bootstrap_draws.npz')
caldraws, helddraws = bd['calibration'], bd['evaluation']
cal = frame[calmask].reset_index(drop=True)
cy = cal.observed_neutrophils_percent.to_numpy(float)
ystar = yy[helddraws]
loss = {}
urows = []
for name in ['locked8', 'marker7', 'official_MCP8']:
    cp = cal['unadapted__' + name].to_numpy()
    hp = held['unadapted__' + name].to_numpy()
    offsets = (cy[caldraws] - cp[caldraws]).mean(axis=1)
    pp = np.clip(hp[helddraws] + offsets[:, None], 0, 100)
    err = pp - ystar
    loss[name] = np.abs(err).mean(axis=1)
    for key, vals in [('MAE', loss[name]), ('R2', 1 - (err * err).sum(axis=1) / ((ystar - ystar.mean(axis=1, keepdims=True)) ** 2).sum(axis=1))]:
        lo, hi = np.quantile(vals, [.025, .975])
        point = next(r[key] for r in rows if r['model'] == name and r['regime'] == 'intercept')
        urows.append(dict(model=name, metric=key, point_estimate=point, CI_low=float(lo), CI_high=float(hi)))
dstar = loss['official_MCP8'] - loss['locked8']
lo, hi = np.quantile(dstar, [.025, .975])
point = next(v['MAE_gain'] for v in contrasts if v['regime'] == 'intercept')
primary = dict(regime='intercept', baseline='official_MCP8', extended='locked8', n=74, MAE_gain=point,
               CI_low=float(lo), CI_high=float(hi), uncertainty='resample_calibration20_and_evaluation74')
contrasts.append(primary)
pd.DataFrame(contrasts).to_csv(OUT / 'paired_MAE_comparisons.csv', index=False)
pd.DataFrame(urows).to_csv(OUT / 'calibration_uncertainty_metrics.csv', index=False)
pd.DataFrame(checks).to_csv(OUT / 'official_implementation_and_fit_checks.csv', index=False)

# Independent scalar-loop check of all2000 two-group bootstrap MAE gains.
loop_gain = []
for i in range(len(caldraws)):
    ym = cy[caldraws[i]]
    ys = yy[helddraws[i]]
    errs = {}
    for name in ['locked8', 'official_MCP8']:
        cp = cal['unadapted__' + name].to_numpy()[caldraws[i]]
        hp = held['unadapted__' + name].to_numpy()[helddraws[i]]
        pp = np.minimum(100, np.maximum(0, hp + (ym.sum() - cp.sum()) / 20))
        errs[name] = sum(abs(float(a) - float(b)) for a, b in zip(ys, pp)) / 74
    loop_gain.append(errs['official_MCP8'] - errs['locked8'])
loopdiff = float(np.max(np.abs(np.array(loop_gain) - dstar)))
assert loopdiff < 1e-10
manifest = dict(date='2026-10-03', exact_official_R_scoring_completed=True,
                same_source125=True, same20_74_roles=True, official_immune_model_dimension=11,
                primary_comparison=primary, robust_conditional_MAE_advantage_gate_passed=bool(primary['CI_low'] > 0),
                max_R_vs_mean_difference=max(r['max_difference'] for r in checks if r['check'].startswith('official_R')),
                source_closed_form_difference=fitdiff, independent_two_group_bootstrap_max_difference=loopdiff,
                general_algorithm_superiority_established=False, new_untouched_external_cohort=False,
                new138_cohort_prediction_executed=False, AD_triage_benefit_established=False,
                original_predictions_unmodified=True, source_parameters_sha256=sha(OUT / 'source_official_MCP8_parameters.json'),
                official_score_file_sha256={scope: sha(BASE / (scope + '_official_MCPcounter_scores.csv')) for scope in ['source', 'target']},
                preserved_before=preserved, preserved_after={n: sha(DATA / n) for n in preserved})
assert preserved == manifest['preserved_after']
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('PRIMARY', primary, 'robust advantage gate', manifest['robust_conditional_MAE_advantage_gate_passed'], flush=True)
