"""Finite paired-CBC pilot. All preprocessing is fit inside each split.

This is exploratory within-study factor prediction, not an AD classifier.
The original module definitions/classifier/manuscript are never overwritten.
"""
import csv
import gzip
import hashlib
import json
import platform
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.stdout.reconfigure(encoding="utf-8")
threadpool_limits(limits=1)
OUT = Path(__file__).resolve().parent
DATA = OUT.parent
PROTOCOL = json.loads((OUT / "execution_protocol.json").read_text(encoding="utf-8"))
SEED = PROTOCOL["seed"]


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


preserved = {p: sha(DATA / p) for p in PROTOCOL["preserve_original"]}
conn = sqlite3.connect((OUT / "raw/Overmyer_2020_09_09.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
clinical = pd.read_sql_query("select * from deidentified_patient_metadata", conn)
rna = pd.read_csv(OUT / "raw/GSE157103_genes.tpm.tsv.gz", sep="\t", index_col=0)
assert rna.index.is_unique and rna.columns.is_unique and clinical.Albany_sampleID.is_unique
assert np.isfinite(rna.to_numpy()).all() and rna.to_numpy().min() >= 0
meta = clinical.set_index("Albany_sampleID")
names = [s for s in rna.columns if s in meta.index]
meta = meta.loc[names].copy()
assert len(meta) == PROTOCOL["expected_paired_n"]
assert meta.sample_id.is_unique

# Verify clinical values through an independently parsed GEO sample-ID join.
geo = {}
with gzip.open(OUT / "raw/GSE157103_series_matrix.txt.gz", "rt", encoding="utf-8") as f:
    rows = [next(csv.reader([line], delimiter="\t")) for line in f if line.startswith("!Sample_")]
get = lambda field: next(r[1:] for r in rows if r[0] == field)
sample_codes, gsms = get("!Sample_description"), get("!Sample_geo_accession")
assert len(set(sample_codes)) == len(sample_codes)
geo = pd.DataFrame({"sample": sample_codes, "GSM": gsms, "title": get("!Sample_title")}).set_index("sample")
# GEO characteristic row positions differ between samples when fields are absent.
# Parse each individual key/value pair, never assume shared row order.
for r in rows:
    if r[0] == "!Sample_characteristics_ch1":
        for sample, value in zip(sample_codes, r[1:]):
            if ":" in value:
                key, text = value.split(":", 1)
                geo.loc[sample, key.strip()] = text.strip()
assert set(rna.columns) == set(geo.index)
meta["GSM"] = geo.loc[names, "GSM"]
assert meta.GSM.is_unique
age_geo = pd.to_numeric(geo.loc[names, "age (years)"], errors="coerce")
age_database = pd.to_numeric(meta.Age_less_than_90, errors="raise")
age_common = age_geo.notna()
age_mismatches = int((age_geo[age_common].to_numpy() != age_database[age_common].to_numpy()).sum())
assert age_mismatches == 0
meta["target_neutrophils_percent"] = pd.to_numeric(meta.Neutrophils_percent, errors="raise")
meta["age"] = pd.to_numeric(meta.Age_less_than_90, errors="raise")
meta["male"] = meta.Gender.map({"M": 1.0, "F": 0.0}).fillna(0.0)
meta["sex_missing"] = (~meta.Gender.isin(["M", "F"])).astype(float)
meta["COVID"] = pd.to_numeric(meta.COVID, errors="raise").astype(int)
assert ((meta.target_neutrophils_percent >= 0) & (meta.target_neutrophils_percent <= 100)).all()
meta.to_csv(OUT / "paired_clinical_metadata.csv", index_label="sample")
exclusions = [dict(sample=s, reason="No exact single clinical record; combined C21 and C54 identity is intentionally unresolved") for s in rna.columns if s not in clinical.Albany_sampleID.to_list()]
pd.DataFrame(exclusions).to_csv(OUT / "RNA_clinical_join_exclusions.csv", index=False)

members = pd.read_csv(DATA / "modules/frozen_pca_modules.csv")
assert len(members) == 5000 and members.gene.is_unique
module_names = sorted(members.module.unique())
available = [g for g in members.gene if g in rna.index]
logrna = np.log2(rna.loc[available, names].to_numpy(float).T + 1.0)
weight = np.zeros((len(available), 8))
lookup = members.set_index("gene")
for j, m in enumerate(module_names):
    flags = lookup.loc[available, "module"].eq(m).to_numpy()
    weight[flags, j] = np.sign(lookup.loc[available, "loading_signed"].to_numpy()[flags]) / members.module.eq(m).sum()
coverage = pd.DataFrame([dict(module=m, original_n=int(members.module.eq(m).sum()), available_n=int(sum(lookup.loc[available, "module"].eq(m)))) for m in module_names])
coverage.to_csv(OUT / "module_gene_coverage.csv", index=False)
markers = pd.read_csv(DATA / "analysis_stage50_triage_bridge/immune_marker_coverage.csv")
markers = markers[markers.score_eligible].copy()
assert len(markers) == 7
marker_values = []
marker_coverage = []
for r in markers.itertuples():
    fixed = r.genes.split(";")
    found = [g for g in fixed if g in rna.index]
    assert len(found) >= 3
    marker_values.append(np.log2(rna.loc[found, names].to_numpy(float) + 1.0).mean(axis=0))
    marker_coverage.append(dict(cell=r.cell, fixed_n=len(fixed), available_n=len(found), genes=";".join(found)))
marker_values = np.stack(marker_values, axis=1)
pd.DataFrame(marker_coverage).to_csv(OUT / "marker_gene_coverage.csv", index=False)
cov = meta[["age", "male", "sex_missing"]].to_numpy(float)
y = meta.target_neutrophils_percent.to_numpy(float)
background = meta.COVID.to_numpy(int)
anchor_idx = [module_names.index(m) for m in ["M01", "M07"]]


def features(train, test, model):
    # Module memberships fixed outside this cohort; RNA z parameters train-only.
    if "frozen8" in model or "M01_M07" in model:
        mean = logrna[train].mean(axis=0)
        sd = logrna[train].std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        a = ((logrna[train] - mean) / sd) @ weight
        b = ((logrna[test] - mean) / sd) @ weight
        if "M01_M07" in model:
            a, b = a[:, anchor_idx], b[:, anchor_idx]
    else:
        a, b = np.empty((len(train), 0)), np.empty((len(test), 0))
    if "marker7" in model:
        a, b = np.column_stack([a, marker_values[train]]), np.column_stack([b, marker_values[test]])
    a, b = np.column_stack([cov[train], a]), np.column_stack([cov[test], b])
    scaler = StandardScaler().fit(a)
    return scaler.transform(a), scaler.transform(b)


def fit_predict(train, test, model, alpha, target=y):
    if model == "training_mean":
        return np.full(len(test), target[train].mean())
    a, b = features(train, test, model)
    fitted = Ridge(alpha=alpha).fit(a, target[train])
    return np.clip(fitted.predict(b), 0.0, 100.0)


def choose_alpha(train, model, stratified=True):
    if model == "training_mean":
        return 0.0
    inner = (StratifiedKFold(4, shuffle=True, random_state=SEED + 1).split(train, background[train])
             if stratified else KFold(4, shuffle=True, random_state=SEED + 1).split(train))
    splits = [(train[a], train[b]) for a, b in inner]
    errors = []
    for alpha in PROTOCOL["ridge_alphas"]:
        errors.append(np.mean([mean_absolute_error(y[b], fit_predict(a, b, model, alpha)) for a, b in splits]))
    return PROTOCOL["ridge_alphas"][int(np.argmin(errors))]


def metrics(actual, pred):
    corr = float(spearmanr(actual, pred).statistic) if np.ptp(actual) > 0 and np.ptp(pred) > 0 else float('nan')
    return dict(R2=float(r2_score(actual, pred)), MAE=float(mean_absolute_error(actual, pred)),
                RMSE=float(np.sqrt(mean_squared_error(actual, pred))), Spearman_r=corr)


indices = np.arange(len(names))
outer_splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(indices, background))
fold_ids = np.full(len(names), -1)
for k, (_, test) in enumerate(outer_splits):
    fold_ids[test] = k
pd.DataFrame({"sample": names, "GSM": meta.GSM, "COVID": background, "outer_fold": fold_ids}).to_csv(OUT / "fixed_outer_folds.csv", index=False)
predictions, tuning = {}, []
for model in PROTOCOL["models"]:
    pred = np.full(len(names), np.nan)
    for fold, (train, test) in enumerate(outer_splits):
        alpha = choose_alpha(train, model)
        pred[test] = fit_predict(train, test, model, alpha)
        tuning.append(dict(model=model, outer_fold=fold, alpha=alpha, n_train=len(train), n_test=len(test)))
    assert np.isfinite(pred).all()
    predictions[model] = pred
    print(model, metrics(y, pred), flush=True)
pd.DataFrame(tuning).to_csv(OUT / "nested_tuning_choices.csv", index=False)
oof = pd.DataFrame({"sample": names, "COVID": background, "observed_neutrophils_percent": y, "outer_fold": fold_ids, **predictions})
oof.to_csv(OUT / "out_of_fold_predictions.csv", index=False)

# Conditional subject bootstrap: resample the same subjects across every model.
rng = np.random.default_rng(SEED + 2)
resamples = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
results = []
for model, pred in predictions.items():
    row = dict(model=model, n=len(y), **metrics(y, pred))
    for key in ["R2", "MAE"]:
        vals = np.array([metrics(y[ix], pred[ix])[key] for ix in resamples])
        row[key + "_CI_low"], row[key + "_CI_high"] = np.quantile(vals, [.025, .975])
    results.append(row)
summary = pd.DataFrame(results)
summary.to_csv(OUT / "prediction_performance.csv", index=False)
contrasts = []
for base, extended in [("age_sex", "age_sex_frozen8"), ("age_sex_marker7", "age_sex_frozen8_marker7"), ("age_sex_marker7", "age_sex_frozen8")]:
    delta = np.abs(y - predictions[base]) - np.abs(y - predictions[extended])
    lo, hi = np.quantile([delta[ix].mean() for ix in resamples], [.025, .975])
    contrasts.append(dict(baseline=base, extended=extended, MAE_reduction=delta.mean(), CI_low=lo, CI_high=hi))
pd.DataFrame(contrasts).to_csv(OUT / "paired_incremental_performance.csv", index=False)

stress, stress_pred = [], []
for train_group, test_group in [(1, 0), (0, 1)]:
    train, test = indices[background == train_group], indices[background == test_group]
    for model in PROTOCOL["models"]:
        alpha = choose_alpha(train, model, stratified=False)
        pred = fit_predict(train, test, model, alpha)
        stress.append(dict(model=model, train_COVID=train_group, test_COVID=test_group, n_train=len(train), n_test=len(test), alpha=alpha, **metrics(y[test], pred)))
        stress_pred.extend(dict(sample=names[i], train_COVID=train_group, model=model, observed=y[i], predicted=p) for i, p in zip(test, pred))
pd.DataFrame(stress).to_csv(OUT / "background_holdout_performance.csv", index=False)
pd.DataFrame(stress_pred).to_csv(OUT / "background_holdout_predictions.csv", index=False)

fixed_model = PROTOCOL["primary_model"]
fixed_features = [features(a, b, fixed_model) for a, b in outer_splits]


def fixed_oof(target):
    result = np.full(len(y), np.nan)
    for (train, test), (a, b) in zip(outer_splits, fixed_features):
        result[test] = np.clip(Ridge(alpha=10.0).fit(a, target[train]).predict(b), 0.0, 100.0)
    return result


fixed_observed_r2 = r2_score(y, fixed_oof(y))
perm_rows = []
for iteration in range(100):
    perm = y.copy()
    for group in [0, 1]:
        mask = indices[background == group]
        perm[mask] = rng.permutation(y[mask])
    perm_rows.append(dict(iteration=iteration, R2=float(r2_score(perm, fixed_oof(perm)))))
pd.DataFrame(perm_rows).to_csv(OUT / "within_background_permutation_null.csv", index=False)
p_perm = (1 + sum(r["R2"] >= fixed_observed_r2 for r in perm_rows)) / 101

# Make a small, ID-linked protein table available, without additional outcome search.
protein = pd.read_sql_query("""
select d.Albany_sampleID as sample, d.sample_id, p.unique_identifier,
       r.rawfile_id, r.batch as protein_TMT_batch, m.metadata_value as gene,
       b.standardized_name as accession, q.raw_abundance, q.normalized_abundance
from proteomics_measurements q
join proteomics_runs p using(replicate_id)
join rawfiles r using(rawfile_id)
join deidentified_patient_metadata d using(sample_id)
join biomolecules b using(biomolecule_id)
join metadata m using(biomolecule_id)
where m.metadata_type='gene_name' and m.metadata_value in ('S100A8','S100A9','ELANE')
""", conn)
protein.to_csv(OUT / "paired_neutrophil_related_proteins_unanalysed.csv", index=False)
conn.close()

primary = next(r for r in results if r["model"] == PROTOCOL["primary_model"])
gate = PROTOCOL["exploratory_pilot_gate"]
passed = (primary["R2"] >= gate["R2_at_least"] and primary["MAE"] <= gate["MAE_at_most_percent_points"]
          and primary["Spearman_r"] >= gate["Spearman_at_least"] and contrasts[0]["CI_low"] > 0)
manifest = dict(
    protocol_sha256=sha(OUT / "execution_protocol.json"),
    dataset="GSE157103 + author MassIVE MSV000085703 SQLite 2020-09-09",
    paired_n=len(names), COVID_n=int(background.sum()), nonCOVID_hospital_n=int((background == 0).sum()),
    neutrophil_target_complete=True, neutrophil_target_range=[float(y.min()), float(y.max())],
    missing_sex_n=int(meta.sex_missing.sum()), age_GEO_numeric_n=int(age_common.sum()), age_GEO_database_mismatches=age_mismatches,
    RNA_only_excluded=exclusions, module_available_genes=len(available), primary_result=primary,
    primary_MAE_increment=contrasts[0], exploratory_pilot_gate_passed=bool(passed),
    fixed_alpha10_observed_R2=float(fixed_observed_r2), within_background_permutation_p=float(p_perm),
    independent_cohort_validation=False, novel_algorithm_established=False,
    paired_protein_rows=len(protein), paired_protein_unique_samples=int(protein[protein['sample'].isin(names)]['sample'].nunique()),
    software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
    preserved_before=preserved, preserved_after={p: sha(DATA / p) for p in preserved},
    input_sha256={p: sha(OUT / p) for p in ['raw/Overmyer_2020_09_09.sqlite', 'raw/GSE157103_genes.tpm.tsv.gz', 'raw/GSE157103_series_matrix.txt.gz']}
)
assert manifest['preserved_before'] == manifest['preserved_after']
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("Pilot gate", bool(passed), "Permutation p", p_perm, "Original inputs unchanged", flush=True)
