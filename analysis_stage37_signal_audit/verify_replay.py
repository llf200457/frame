from pathlib import Path
import json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parent
names=['predictions.csv','training_log.json','inner_splits.json','quality_covariates.csv','combined_predictions.csv','region_metrics.csv','macro_summary.csv','paired_intervals.csv','residual_records.csv','residual_associations.csv']
before={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
(ROOT/'replay_before_hashes.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
for script in ['run_quality_audit.py','analyze_audit.py']:
    with (ROOT/(script+'.replay.log')).open('w',encoding='utf-8') as log: subprocess.run([sys.executable,str(ROOT/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
after={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
result={'full_new_180_fit_retraining':True,'files':{n:{'before':before[n],'after':after[n],'identical':before[n]==after[n]} for n in names}}
(ROOT/'replay_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
assert all(v['identical'] for v in result['files'].values())
print('Full retraining and analysis replay: all 10 files byte-identical.',flush=True)
