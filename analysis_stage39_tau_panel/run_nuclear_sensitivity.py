from pathlib import Path
import json,hashlib
import pandas as pd
from run_panel import dataset,nested
ROOT=Path(__file__).resolve().parent
donors,cov,genes,ann,mi,ys,targets=dataset();filtered=[];annotations=[]
for g,a in zip(genes,ann):
 keep=~a['index'].str.startswith('MT-',na=False).to_numpy();filtered.append(g[:,keep]);annotations.append(a.loc[keep].reset_index(drop=True))
output=[];logs=[]
for target in ['Tau_panel','Tau_DFC']:
 for seed in [2026,2027,2028]:
  o,l=nested(donors,cov,filtered,annotations,[[],[]],ys[:,targets.index(target)],target,seed,['genes']);output+=o;logs+=l
pd.DataFrame(output).to_csv(ROOT/'nuclear_sensitivity_predictions.csv',index=False)
(ROOT/'nuclear_sensitivity_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
(ROOT/'nuclear_sensitivity_manifest.json').write_text(json.dumps({'outer_fits':len(logs),'excluded':'symbol prefix MT- before variance/correlation selection; original all-gene CPM denominator retained','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
