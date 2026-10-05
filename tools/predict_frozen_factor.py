"""Portable arithmetic scoring of the protected neutrophil percentage predictor."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'workflow/analysis_stage53_measured_blood_anchor/locked_neutrophil_factor_predictor.npz'
def score(tpm,meta,model,offset=0.):
    assert tpm.index.is_unique and tpm.columns.is_unique,'Gene/sample names must be unique'
    assert meta['sample'].is_unique and len(meta)==len(tpm.columns),'Metadata samples must be unique'
    assert set(meta['sample'])==set(tpm.columns),'TPM/metadata sample sets differ'
    names=meta['sample'].tolist();expr=tpm.loc[:,names]
    values=expr.to_numpy(float)
    assert np.isfinite(values).all() and (values>=0).all(),'TPM must be finite and nonnegative'
    cov=meta[['age','male','sex_missing']].to_numpy(float)
    assert np.isfinite(cov).all() and ((cov[:,0]>=0)&(cov[:,0]<=120)).all(),'Invalid covariates'
    assert np.isin(cov[:,1:],[0,1]).all(),'male/sex_missing must be binary'
    assert np.isfinite(offset),'Offset must be finite'
    genes=model['genes'].tolist();available=[g for g in genes if g in expr.index]
    assert len(available)/len(genes)>=.9,'Fewer than 90% of fixed genes are available'
    lookup={g:i for i,g in enumerate(genes)};ix=np.array([lookup[g] for g in available],dtype=int)
    z=np.zeros((len(names),len(genes)))
    z[:,ix]=(np.log2(expr.loc[available].to_numpy(float).T+1)-model['gene_mean'][ix])/model['gene_sd'][ix]
    modules=z@model['module_weights']
    x=np.column_stack([cov,modules])
    raw=np.clip(((x-model['feature_mean'])/model['feature_scale'])@model['coefficients']+model['intercept'],0.,100.)
    out=pd.DataFrame({'sample':names,'raw_neutrophils_percent':raw})
    if offset!=0:out['offset_neutrophils_percent']=np.clip(raw+offset,0.,100.)
    return out,{'model_genes':len(genes),'available_genes':len(available),'missing_genes':len(genes)-len(available),
        'samples':len(names),'offset':offset,'retrained':False,'target_gene_rescaling':False}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tpm',required=True,type=Path);ap.add_argument('--metadata',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--model',type=Path,default=DEFAULT)
    ap.add_argument('--offset',type=float,default=0.)
    a=ap.parse_args()
    if a.output.exists():ap.error('Output exists; choose a new filename')
    if a.model.resolve()==DEFAULT.resolve():
        assert hashlib.sha256(a.model.read_bytes()).hexdigest()=='1cf23a1a82e1adb4fef00fa6b56d1bef52cf2bd11c6847f7c4bebc33a9ecc424','Protected default model differs'
    sep=',' if a.tpm.name.endswith(('.csv','.csv.gz')) else '\t'
    tpm=pd.read_csv(a.tpm,sep=sep,index_col=0)
    meta=pd.read_csv(a.metadata,dtype={'sample':str})
    with np.load(a.model,allow_pickle=False) as model:out,report=score(tpm,meta,model,a.offset)
    a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False)
    report['model_sha256']=hashlib.sha256(a.model.read_bytes()).hexdigest()
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
