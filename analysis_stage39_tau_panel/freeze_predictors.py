from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from run_panel import dataset,design,predict
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'frozen_models';OUT.mkdir(exist_ok=True)
donors,cov,genes,ann,mi,ys,targets=dataset();allrows=np.arange(len(donors));covnames=pd.read_csv(ROOT/'analysis_cohort.csv').columns[1:31].tolist();cards=[]
for target in ['Tau_panel','Tau_DFC']:
 y=ys[:,targets.index(target)];inner=list(KFold(3,shuffle=True,random_state=2026).split(donors));packs=[(iv,design(cov,genes,mi,y,it,iv,'genes')) for it,iv in inner];grid=[.01,.1,1.];scores=[float(np.mean([np.mean(abs(predict(pack,a,'genes')[0]-y[iv])) for iv,pack in packs])) for a in grid];alpha=grid[int(np.argmin(scores))];ref=design(cov,genes,mi,y,allrows,allrows,'genes');expected,model=predict(ref,alpha,'genes');selected=ref[-1]
 imp=SimpleImputer().fit(cov);cs=StandardScaler().fit(imp.transform(cov));parts=[cs.transform(imp.transform(cov))];sel=[]
 for cell,g,a,ii in zip(['Astrocyte','Micro-PVM'],genes,ann,selected):
  parts.append(g[:,ii]);sel.append({'cell':cell,'gene_ids':a.gene_ids.iloc[ii].tolist(),'symbols':a['index'].iloc[ii].tolist()})
 x=np.column_stack(parts);sc=StandardScaler().fit(x);pred=model.predict(sc.transform(x))*y.std()+y.mean();assert np.allclose(pred,expected,rtol=0,atol=1e-12)
 bundle={'target':target,'covariate_names':covnames,'covariate_imputer':imp,'covariate_scaler':cs,'selected_genes':sel,'joint_scaler':sc,'target_mean':float(y.mean()),'target_sd':float(y.std()),'model':model,'input_expression':'log2(1+CPM) with total all-gene counts in each cell-class denominator','input_region':'MTG','training_donors':donors.tolist()};joblib.dump(bundle,OUT/(target+'.joblib'))
 card={'target':target,'training_donors':len(donors),'alpha':alpha,'inner_mae':scores,'selected_genes':sel,'scope':'Research only. Fully refitted frozen predictor; training performance is not validation. Held-donor results are in the original OOF files. No independent AT8-cohort validation.','predictor_sha256':hashlib.sha256((OUT/(target+'.joblib')).read_bytes()).hexdigest(),'software':'Python3.13.11/scikit-learn1.9.1; joblib bundles require compatible runtime','primary_target_unchanged':True,'reconstructed_training_prediction_max_error':float(abs(pred-expected).max())};cards.append(card);(OUT/(target+'_model_card.json')).write_text(json.dumps(card,indent=2),encoding='utf-8')
(OUT/'freeze_manifest.json').write_text(json.dumps({'models':cards,'seed':2026,'final_inner_fits':18,'final_full_donor_fits':2,'not_counted_as_validation':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8');print('Two research predictors frozen; reconstruction verified, no training accuracy reported as validation.')
