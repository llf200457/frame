"""Finite measured-CBC direction audit; no source model or classifier refit.

RNA-to-array direction transfer is explicitly exploratory and produces no CBC
percentage on arrays. Mathematical support decomposition is not causal biology.
"""
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from scipy.stats import rankdata, spearmanr, t
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding='utf-8')
threadpool_limits(limits=1)
OUT = Path(__file__).resolve().parent
DATA = OUT.parent
S53 = DATA / 'analysis_stage53_measured_blood_anchor'
S54 = DATA / 'analysis_stage54_soundlife_external'
S51 = DATA / 'analysis_stage51_cell_conditioned_transport'
S50 = DATA / 'analysis_stage50_triage_bridge'
PROTO = json.loads((OUT / 'execution_protocol.json').read_text(encoding='utf-8'))
SEED = PROTO['seed']
BOOT = PROTO['bootstrap_n']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


preserved = {name: sha(DATA / name) for name in PROTO['preserve_files']}
par = json.loads((DATA / 'metadata/stage22/primary_classifier_parameters.json').read_text(encoding='utf-8'))
locked = np.load(S53 / 'locked_neutrophil_factor_predictor.npz', allow_pickle=False)
assert sha(S53 / 'locked_neutrophil_factor_predictor.npz') == '1cf23a1a82e1adb4fef00fa6b56d1bef52cf2bd11c6847f7c4bebc33a9ecc424'
mods = par['features']
assert locked['feature_names'][3:].tolist() == mods
b_ad = np.asarray(par['coefficients'][0]) / np.asarray(par['scaler_scale'])
b_neut = locked['coefficients'][3:] / locked['feature_scale'][3:]
assert np.isfinite(b_ad).all() and np.isfinite(b_neut).all()
pd.DataFrame({'module': mods, 'original_AD_raw_module_coefficient': b_ad,
              'measured_neutrophil_raw_module_direction': b_neut}).to_csv(
    OUT / 'fixed_module_directions.csv', index=False)

# Reconstruct source RNA modules without fitting a new transform.
source_meta = pd.read_csv(S53 / 'paired_clinical_metadata.csv', index_col='sample')
source_expr = pd.read_csv(S53 / 'raw/GSE157103_genes.tpm.tsv.gz', sep='\t', index_col=0)
genes = locked['genes'].tolist()
z = (np.log2(source_expr.loc[genes, source_meta.index].to_numpy(float).T + 1) - locked['gene_mean']) / locked['gene_sd']
source_modules = z @ locked['module_weights']
source = pd.DataFrame(source_modules, columns=mods, index=source_meta.index)
source['sample'] = source.index
source['cohort'] = 'Overmyer125'
source['measured_neutrophils_percent'] = source_meta.target_neutrophils_percent.to_numpy(float)
for name in ['age', 'male', 'sex_missing', 'COVID']:
    source[name] = source_meta[name].to_numpy(float)
source['analysis_role'] = 'source_training_association'

target_pred = pd.read_csv(S54 / 'primary_one_visit_predictions.csv', index_col='sample')
target_modules = pd.read_csv(S54 / 'external_module_scores.csv', index_col='sample').loc[target_pred.index, mods]
target = target_modules.copy()
target['sample'] = target.index
target['cohort'] = 'SoundLife94'
target['measured_neutrophils_percent'] = target_pred.observed_neutrophils_percent.to_numpy(float)
target['age'] = target_pred.age.to_numpy(float)
target['male'] = target_pred.male.to_numpy(float)
target['sex_missing'] = 0.0
target['COVID'] = 0.0
target['analysis_role'] = 'previously_evaluated_independent_healthy_cohort'
target['subject'] = target_pred.subject.to_numpy()
assert target.subject.is_unique

for frame in [source, target]:
    module_values = frame[mods].to_numpy(float)
    frame['AD_weighted_RNA_projection'] = (module_values - locked['feature_mean'][3:]) @ b_ad
    frame['neutrophil_module_only_projection'] = (module_values - locked['feature_mean'][3:]) @ b_neut
    # The covariate terms and intercept are intentionally omitted from these
    # expression-only projections. Neither projection is a clinical AD score.
pd.concat([source, target], ignore_index=True).to_csv(OUT / 'RNA_measured_factor_projection_records.csv', index=False)


def partial_rank(a, b, cov):
    ra, rb = rankdata(a), rankdata(b)
    ar = ra - cov @ np.linalg.lstsq(cov, ra, rcond=None)[0]
    br = rb - cov @ np.linalg.lstsq(cov, rb, rcond=None)[0]
    if np.std(ar) < 1e-12 or np.std(br) < 1e-12:
        return np.nan, np.nan, np.linalg.matrix_rank(cov)
    rho = float(np.corrcoef(ar, br)[0, 1])
    rank = int(np.linalg.matrix_rank(cov))
    df = len(a) - rank - 1
    p = float(2 * t.sf(abs(rho) * np.sqrt(df / max(1e-15, 1 - rho * rho)), df))
    return rho, p, rank


def covariates(frame, diagnosis=False, expanded=False):
    columns = [np.ones(len(frame)), rankdata(frame.age.to_numpy(float)),
               frame.male.to_numpy(float)]
    if 'sex_missing' in frame:
        columns.append(frame.sex_missing.to_numpy(float))
    if diagnosis:
        columns.append(frame.status.eq('AD').to_numpy(float))
    if expanded:
        columns.extend([frame.apoe_e4_count.to_numpy(float), frame.apoe_missing.to_numpy(float)])
        dummy = pd.get_dummies(frame['batch'].astype(str), drop_first=True, dtype=float)
        columns.extend(dummy.to_numpy().T)
    return np.column_stack(columns)


def boot_partial(a, b, frame, rng, source_background=False, diagnosis=False, expanded=False, groups=None):
    values = []
    for _ in range(BOOT):
        if groups is None:
            take = rng.integers(0, len(frame), len(frame))
        else:
            take = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in groups])
        draw = frame.iloc[take]
        c = covariates(draw, diagnosis=diagnosis, expanded=expanded)
        if source_background:
            c = np.column_stack([c, draw.COVID.to_numpy(float)])
        r, _, _ = partial_rank(a[take], b[take], c)
        if np.isfinite(r):
            values.append(r)
    assert len(values) >= int(BOOT * .95)
    return np.quantile(values, [.025, .975]), len(values)


def bh(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    sorted_q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty(len(p))
    q[order] = np.minimum(sorted_q, 1)
    return q


measure_rows = []
rng = np.random.default_rng(SEED + 55)
for frame in [source, target]:
    c = covariates(frame)
    is_source = frame.cohort.iloc[0] == 'Overmyer125'
    if is_source:
        c = np.column_stack([c, frame.COVID.to_numpy(float)])
    y = frame.measured_neutrophils_percent.to_numpy(float)
    for index in ['AD_weighted_RNA_projection', 'neutrophil_module_only_projection', 'M01', 'M07']:
        a = frame[index].to_numpy(float)
        r, p, rank = partial_rank(a, y, c)
        ci, nboot = boot_partial(a, y, frame, rng, source_background=is_source)
        measure_rows.append(dict(cohort=frame.cohort.iloc[0], projection=index, n=len(frame),
                                 role=frame.analysis_role.iloc[0], raw_Spearman=float(spearmanr(a, y).statistic),
                                 partial_rank_r=r, p=p, CI_low=ci[0], CI_high=ci[1],
                                 covariate_rank=rank, valid_bootstrap_n=nboot,
                                 comparison_family='primary_fixed_projections' if index.endswith('projection') else 'module_diagnostic'))
measure = pd.DataFrame(measure_rows)
measure['q_BH'] = np.nan
for _, indices in measure.groupby('comparison_family').groups.items():
    measure.loc[indices, 'q_BH'] = bh(measure.loc[indices, 'p'])
measure.to_csv(OUT / 'RNA_measured_factor_associations.csv', index=False)
print('Measured-factor links:', measure[['cohort', 'projection', 'partial_rank_r', 'q_BH']].to_string(index=False), flush=True)

# The support geometry is original discovery-array geometry, not RNA geometry.
modules = {c: pd.read_csv(S51 / f'{c}_frozen_module_input.csv', index_col=0)[mods]
           for c in ['GSE63060', 'GSE63061', 'GSE140829']}
reference = modules['GSE63060'].to_numpy(float)
assert len(reference) == 330
location = reference.mean(axis=0)
covariance = np.cov(reference, rowvar=False)
precision = np.linalg.pinv(covariance)
assert np.linalg.matrix_rank(covariance) == 8
direction_variance = float(b_neut @ covariance @ b_neut)
assert direction_variance > 0


def support(frame):
    delta = frame[mods].to_numpy(float) - location
    original = np.einsum('ij,jk,ik->i', delta, precision, delta)
    index = (delta @ b_neut) / np.sqrt(direction_variance)
    immune = index ** 2
    residual = original - immune
    assert residual.min() > -1e-7
    return original, index, immune, np.maximum(residual, 0)


original, index, immune, residual = support(modules['GSE63060'])
thresholds = {'original_module_drift': float(np.quantile(original, .95)),
              'measurement_anchored_immune_distance': float(np.quantile(immune, .95)),
              'measurement_anchored_residual_distance': float(np.quantile(residual, .95))}
np.savez_compressed(OUT / 'anchored_support_reference.npz', mean=location, covariance=covariance,
                    precision=precision, measured_direction=b_neut,
                    direction_variance=np.asarray(direction_variance), module_names=np.asarray(mods))
pd.DataFrame({'sample': modules['GSE63060'].index, 'original_module_drift': original,
              'signed_immune_direction_index': index, 'measurement_anchored_immune_distance': immune,
              'measurement_anchored_residual_distance': residual}).to_csv(OUT / 'all330_source_support_reference.csv', index=False)

previous = pd.read_csv(S50 / 'triage_immune_audit_predictions.csv')
metadata0 = pd.read_csv(DATA / 'processed/GSE63060_matrix_formal_labels.csv', index_col='matrix_sample')
metadata1 = pd.read_csv(DATA / 'metadata/stage20/GSE63061_relabelled_predictions_all_samples.csv', index_col='matrix_sample')
metadata2 = pd.read_csv(DATA / 'analysis_stage26/GSE140829_linked_metadata.csv', index_col='expression_id')
series = (DATA / 'metadata/GSE63061_series_metadata.txt').read_text(encoding='utf-8')
fields = [next(csv.reader([line], delimiter='\t')) for line in series.splitlines() if line.startswith('!Sample_')]
gsm = next(v[1:] for v in fields if v[0] == '!Sample_geo_accession')
for name in ['age', 'gender']:
    values = next(v[1:] for v in fields if v[0] == '!Sample_characteristics_ch1' and any(s.startswith(name + ':') for s in v[1:]))
    mapping = dict(zip(gsm, [v.split(':', 1)[1].strip() for v in values]))
    metadata1[name] = metadata1.geo_accession.map(mapping)

records, fullmetrics, gate_rows, drift_rows, gain_rows = [], [], [], [], []
decomposition_errors = []


def accept(score, coverage):
    # Frames are sorted by exact sample ID before stable ordering.
    return np.argsort(score, kind='stable')[:int(np.floor(len(score) * coverage))]


for cohort in modules:
    frame = previous[previous.cohort.eq(cohort)].copy().sort_values('sample').reset_index(drop=True)
    m = modules[cohort].loc[frame['sample']]
    frozen_logit = ((m.to_numpy() - np.asarray(par['scaler_mean'])) / np.asarray(par['scaler_scale'])) @ np.asarray(par['coefficients'][0]) + par['intercept'][0]
    p = expit(frozen_logit)
    assert np.abs(p - frame.p_frozen.to_numpy()).max() < 1e-10
    original, index, immune, residual = support(m)
    assert np.abs(original - frame.module_drift.to_numpy()).max() < 1e-8
    decomposition_errors.append(float(np.abs(original - immune - residual).max()))
    frame['signed_immune_direction_index'] = index
    frame['original_module_drift'] = original
    frame['measurement_anchored_immune_distance'] = immune
    frame['measurement_anchored_residual_distance'] = residual
    frame['frozen_prediction_uncertainty'] = 1 - np.abs(2 * p - 1)
    frame['immune_geometric_share'] = np.divide(immune, original, out=np.zeros_like(immune), where=original > 1e-14)
    if cohort == 'GSE63060':
        meta = metadata0.loc[frame['sample']]
        age, sex = meta['age:ch1'], meta['gender:ch1']
        assert meta.label_formal.replace({'Control': 'CTL'}).to_list() == frame.status.to_list()
    elif cohort == 'GSE63061':
        meta = metadata1.loc[frame['sample']]
        age, sex = meta.age, meta.gender
        assert meta.status.to_list() == frame.status.to_list()
    else:
        meta = metadata2.loc[frame['sample']]
        age, sex = meta.age_at_draw, meta.sex
        assert meta.diagnosis.replace({'Control': 'CTL'}).to_list() == frame.status.to_list()
        frame['batch'] = meta.batch.to_numpy()
        genotype = meta.apoe_status.fillna('').astype(str)
        valid_genotype = genotype.str.fullmatch(r'E[234]_E[234]')
        frame['apoe_e4_count'] = np.where(valid_genotype, genotype.str.count('E4').to_numpy(float), 0.)
        frame['apoe_missing'] = (~valid_genotype).to_numpy(float)
        frame['apoe_genotype'] = genotype.to_numpy()
    frame['age'] = pd.to_numeric(age, errors='coerce').to_numpy(float)
    male = sex.astype(str).str.lower().isin(['male', 'm'])
    female = sex.astype(str).str.lower().isin(['female', 'f'])
    frame['male'] = male.to_numpy(float)
    frame['sex_missing'] = (~(male | female)).to_numpy(float)
    y = frame.status.eq('AD').to_numpy(int)
    loss = (p - y) ** 2
    error = (p >= .5) != y
    assert np.abs(loss - frame.squared_error.to_numpy()).max() < 1e-10
    fullmetrics.append(dict(cohort=cohort, n=len(frame), AD=int(y.sum()), controls=int((y == 0).sum()),
                            original_brier=float(loss.mean()), original_hard_error_rate=float(error.mean()),
                            age_missing=int(frame.age.isna().sum()), sex_missing=int(frame.sex_missing.sum()),
                            unknown_donor_independence_not_resolved=True))
    for group in ['all', 'AD', 'CTL']:
        take = np.ones(len(frame), bool) if group == 'all' else frame.status.eq(group).to_numpy()
        drift_rows.append(dict(cohort=cohort, status=group, n=int(take.sum()),
                               median_original_drift=float(np.median(original[take])),
                               median_immune_distance=float(np.median(immune[take])),
                               median_residual_distance=float(np.median(residual[take])),
                               median_immune_geometric_share=float(np.median(frame.immune_geometric_share.to_numpy()[take]))))
    score_names = PROTO['comparators']
    for name in score_names:
        score = frame[name].to_numpy(float)
        if name in thresholds:
            retained = score <= thresholds[name]
            gate_rows.append(dict(cohort=cohort, score=name, threshold=thresholds[name],
                                  n=len(frame), retained_n=int(retained.sum()), coverage=float(retained.mean()),
                                  retained_brier=float(loss[retained].mean()) if retained.any() else np.nan,
                                  retained_error_rate=float(error[retained].mean()) if retained.any() else np.nan))
        for coverage in PROTO['fixed_coverages']:
            selected = accept(score, coverage)
            fullmetrics.append(dict(cohort=cohort, score=name, target_coverage=coverage, n=len(frame),
                                    retained_n=len(selected), retained_AD=int(y[selected].sum()),
                                    retained_CTL=int((y[selected] == 0).sum()),
                                    retained_brier=float(loss[selected].mean()),
                                    retained_error_rate=float(error[selected].mean()),
                                    actual_coverage=len(selected) / len(frame)))
    if cohort != 'GSE63060':
        classes = [np.flatnonzero(y == k) for k in [0, 1]]
        rng_gain = np.random.default_rng(SEED + 155)
        # Same resampled IDs across candidate/baselines/coverage values.
        draws = [np.concatenate([rng_gain.choice(ids, len(ids), replace=True) for ids in classes]) for _ in range(BOOT)]
        candidate = frame.measurement_anchored_residual_distance.to_numpy(float)
        for coverage in PROTO['fixed_coverages']:
            for baseline in ['original_module_drift', 'frozen_prediction_uncertainty']:
                comparator = frame[baseline].to_numpy(float)
                observed = float(loss[accept(comparator, coverage)].mean() - loss[accept(candidate, coverage)].mean())
                gains = [float(loss[idx][accept(comparator[idx], coverage)].mean() - loss[idx][accept(candidate[idx], coverage)].mean()) for idx in draws]
                lo, hi = np.quantile(gains, [.025, .975])
                gain_rows.append(dict(cohort=cohort, target_coverage=coverage, baseline=baseline,
                                      candidate='measurement_anchored_residual_distance',
                                      brier_gain=observed, CI_low=float(lo), CI_high=float(hi),
                                      conditional_stratified_bootstrap_n=BOOT))
    records.append(frame)

joined = pd.concat(records, ignore_index=True)
joined.to_csv(OUT / 'all_AD_array_audit_records.csv', index=False)
pd.DataFrame(fullmetrics).to_csv(OUT / 'support_selection_performance.csv', index=False)
pd.DataFrame(gate_rows).to_csv(OUT / 'fixed_source_gate_performance.csv', index=False)
pd.DataFrame(drift_rows).to_csv(OUT / 'drift_partition_summary.csv', index=False)
gains = pd.DataFrame(gain_rows)
gains.to_csv(OUT / 'paired_retained_Brier_gains.csv', index=False)

# Error associations are explanatory, stratified by diagnosis; they do not fit
# a new prediction rule and cannot identify a causal effect of cell composition.
error_rows = []
rng_error = np.random.default_rng(SEED + 255)
for cohort in modules:
    for expanded in [False, True] if cohort == 'GSE140829' else [False]:
        for group in ['AD', 'CTL', 'all']:
            frame = joined[joined.cohort.eq(cohort)].copy()
            if group != 'all':
                frame = frame[frame.status.eq(group)].copy()
            frame = frame[frame.age.notna()].reset_index(drop=True)
            c = covariates(frame, diagnosis=group == 'all', expanded=expanded)
            a = frame.signed_immune_direction_index.to_numpy(float)
            b = frame.squared_error.to_numpy(float)
            r, p, rank = partial_rank(a, b, c)
            # Expanded inference retains observed batch/diagnosis stratum sizes.
            strata = None
            if expanded:
                strata = [np.asarray(v, int) for v in frame.groupby(['batch', 'status'], sort=True).groups.values()]
            ci, nboot = boot_partial(a, b, frame, rng_error, diagnosis=group == 'all', expanded=expanded, groups=strata)
            error_rows.append(dict(cohort=cohort, status=group, n=len(frame),
                                   adjustment='age_sex_APOE_batch' if expanded else 'age_sex',
                                   diagnosis_adjusted=group == 'all',
                                   partial_rank_r=r, p=p, CI_low=float(ci[0]), CI_high=float(ci[1]),
                                   covariate_rank=rank, valid_bootstrap_n=nboot,
                                   bootstrap_scheme='within_batch_and_status' if expanded else 'sample',
                                   source_discovery_descriptive_only=cohort == 'GSE63060'))
error_asso = pd.DataFrame(error_rows)
error_asso['q_BH_all12'] = bh(error_asso.p)
error_asso.to_csv(OUT / 'immune_direction_error_associations.csv', index=False)
joined[joined.cohort.eq('GSE140829')].groupby(['batch', 'status']).agg(
    n=('sample', 'size'), APOE_missing=('apoe_missing', 'sum')).reset_index().to_csv(
        OUT / 'independent_batch_genotype_audit.csv', index=False)

primary = gains[gains.target_coverage.eq(.8)]
passed = bool(len(primary) == 4 and (primary.CI_low > 0).all())
dependencies = [
    S53 / 'locked_neutrophil_factor_predictor.npz', S53 / 'paired_clinical_metadata.csv',
    S53 / 'raw/GSE157103_genes.tpm.tsv.gz', S54 / 'primary_one_visit_predictions.csv',
    S54 / 'external_module_scores.csv', S50 / 'triage_immune_audit_predictions.csv',
    DATA / 'processed/GSE63060_matrix_formal_labels.csv',
    DATA / 'metadata/stage20/GSE63061_relabelled_predictions_all_samples.csv',
    DATA / 'metadata/GSE63061_series_metadata.txt', DATA / 'analysis_stage26/GSE140829_linked_metadata.csv',
] + [S51 / f'{cohort}_frozen_module_input.csv' for cohort in modules]
manifest = dict(date=PROTO['date'], seed=SEED, protocol_sha256=sha(OUT / 'execution_protocol.json'),
                script_sha256=sha(Path(__file__)), all_source_files_sha256={str(p.relative_to(DATA)): sha(p) for p in dependencies},
                preserved_before=preserved, preserved_after={name: sha(DATA / name) for name in preserved},
                independent_RNA_subjects=94, RNA_source_subjects=125,
                AD_array_profiles_by_cohort=joined.groupby('cohort').size().to_dict(),
                original_covariance_reference_profiles=330,
                covariance_rank=int(np.linalg.matrix_rank(covariance)),
                decomposition_max_absolute_error=max(decomposition_errors),
                fixed_source_thresholds=thresholds,
                primary_four_contrast_exploratory_gate_passed=passed,
                original_classifier_changed=False, measured_predictor_changed=False,
                external_labels_used_for_direction_or_score_fit=False,
                previously_evaluated_external_cohorts=True,
                inferred_measured_CBC_on_AD_arrays=False, causal_immune_error_claim=False,
                new_general_algorithm_established=False, clinical_deployment_established=False,
                bootstrap_conditioned_on_source_weights_and_reference=True,
                software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                              scipy=scipy.__version__, sklearn=sklearn.__version__))
assert manifest['preserved_before'] == manifest['preserved_after']
(OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Primary finite gate:', passed, flush=True)
print(primary.to_string(index=False), flush=True)
print(error_asso[['cohort', 'status', 'adjustment', 'partial_rank_r', 'q_BH_all12']].to_string(index=False), flush=True)
