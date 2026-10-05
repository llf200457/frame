"""Offline integrity and SAVED aggregate-value correspondence audit, stdlib only."""
import argparse,csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.parse_args()
    records=rows(ROOT/'SHA256SUMS.csv')
    for r in records:
        p=ROOT/r['file'];assert p.is_file(),r['file']
        assert p.stat().st_size==int(r['bytes']),r['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['file']
    hashes=json.loads((ROOT/'docs/protected_model_hashes.json').read_text())
    for name,h in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    d=ROOT/'results/blood_anchor'
    numbers=json.loads((d/'manuscript_numbers.json').read_text())
    source=next(r for r in rows(ROOT/'results/source_prediction/prediction_performance.csv') if r['model']=='age_sex_frozen8')
    official=rows(d/'official_performance.csv')
    raw=next(r for r in official if r['model']=='locked8' and r['regime']=='unadapted')
    offset=next(r for r in official if r['model']=='locked8' and r['regime']=='intercept')
    mcp=next(r for r in official if r['model']=='official_MCP8' and r['regime']=='intercept')
    uncertainty=rows(d/'two_group_uncertainty.csv')
    checks={'SourceRtwo':float(source['R2']),'SourceMAE':float(source['MAE']),'SourceRho':float(source['Spearman_r']),
        'TargetRawRtwo':float(raw['R2']),'TargetRawMAE':float(raw['MAE']),'TargetRawBias':float(raw['bias']),
        'TargetRtwo':float(offset['R2']),'TargetMAE':float(offset['MAE']),'TargetRho':float(offset['Spearman_r']),
        'MCPGain':float(mcp['MAE'])-float(offset['MAE'])}
    aucs={'GSE63061':'RelatedAUC','GSE140829':'IndependentAUC','GSE249477':'RNAFirstAUC','GSE248417':'RNASecondAUC'}
    for r in rows(d/'clinical_transport.csv'):checks[aucs[r['cohort']]]=float(r['AUC'])
    # Additional interval endpoints are saved values, not bootstrap recomputation.
    for r in uncertainty:
        if r['model']=='locked8' and r['metric']=='R2':
            checks.update(TargetRtwoLow=float(r['CI_low']),TargetRtwoHigh=float(r['CI_high']))
    contrast=next(r for r in rows(d/'official_comparison.csv') if r['baseline']=='official_MCP8'
        and r['regime']=='intercept' and r['uncertainty']=='resample_calibration20_and_evaluation74')
    checks.update(MCPGainLow=float(contrast['CI_low']),MCPGainHigh=float(contrast['CI_high']))
    for k,v in checks.items():assert math.isfinite(v) and abs(v-numbers[k])<1e-10,(k,v,numbers[k])
    assert len(checks)==18,('saved-value checks',len(checks))
    print(json.dumps({'passed':True,'files_hashed':len(records),'protected_models':len(hashes),
        'saved_value_matches':len(checks),'raw_patient_metrics_recomputed':False,
        'bootstrap_intervals_recomputed':False,'model_training_executed':False},indent=2))
if __name__=='__main__':main()
