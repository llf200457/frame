from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_panel import dataset,nested
ROOT=Path(__file__).resolve().parent
donors,cov,genes,ann,mi,ys,targets=dataset();rng=np.random.default_rng(39002);rows=pd.read_csv(ROOT/'permutation_summary.csv').to_dict('records') if (ROOT/'permutation_summary.csv').exists() else [];logs=json.loads((ROOT/'permutation_training_log.json').read_text()) if (ROOT/'permutation_training_log.json').exists() else []
done=len(rows)
for repeat in range(199):
 y=rng.permutation(ys[:,0])
 if repeat<done:continue
 o,l=nested(donors,cov,genes,ann,mi,y,'Tau_panel_permuted',2026,['metadata','genes']);z=pd.DataFrame(o);e=z.assign(error=abs(z.observed-z.predicted)).groupby('model').error.mean()
 rows.append(dict(repeat=repeat,metadata_mae=float(e['metadata']),genes_mae=float(e['genes']),improvement=float(e['metadata']-e['genes'])));logs.extend([dict(permutation=repeat,**v) for v in l]);pd.DataFrame(rows).to_csv(ROOT/'permutation_summary.csv',index=False)
(ROOT/'permutation_training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
