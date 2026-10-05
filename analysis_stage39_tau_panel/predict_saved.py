"""Inference on already normalized, matched MTG features; never accepts outcome columns."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parent
def predict_saved(bundle,covariates,expression):
    # expression: cell-class -> DataFrame, rows=donor IDs, columns=Ensembl gene IDs,
    # values=log2(1+CPM). Requires compatible all-gene CPM denominator upstream.
    cov=covariates.loc[:,bundle['covariate_names']]
    parts=[bundle['covariate_scaler'].transform(bundle['covariate_imputer'].transform(cov.to_numpy(float)))]
    for entry in bundle['selected_genes']:
        frame=expression[entry['cell']].loc[cov.index,entry['gene_ids']]
        if not np.isfinite(frame.to_numpy()).all():raise ValueError('Missing/nonfinite selected gene values')
        parts.append(frame.to_numpy(float))
    x=bundle['joint_scaler'].transform(np.column_stack(parts))
    return pd.Series(bundle['model'].predict(x)*bundle['target_sd']+bundle['target_mean'],index=cov.index,name=bundle['target'])
if __name__=='__main__':
    from run_panel import dataset,design,predict
    donors,cov,g,ann,mi,ys,targets=dataset();rows=np.arange(len(donors));audit=[]
    for target in ['Tau_panel','Tau_DFC']:
        bundle=joblib.load(ROOT/'frozen_models'/(target+'.joblib'));covframe=pd.DataFrame(cov,index=donors,columns=bundle['covariate_names']);expr={cell:pd.DataFrame(v,index=donors,columns=a.gene_ids) for cell,v,a in zip(['Astrocyte','Micro-PVM'],g,ann)}
        actual=predict_saved(bundle,covframe,expr).to_numpy();expected,_=predict(design(cov,g,mi,ys[:,targets.index(target)],rows,rows,'genes'),bundle['model'].alpha,'genes');err=float(abs(actual-expected).max());assert err<1e-12
        audit.append({'target':target,'saved_model_reload_max_error':err,'validated_inference_implementation_only':True,'independent_validation':False})
    (ROOT/'frozen_models'/'inference_check.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
