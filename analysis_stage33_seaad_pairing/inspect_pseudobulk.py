from pathlib import Path
import json
import h5py
import pandas as pd
ROOT=Path(__file__).resolve().parent
def strings(ds):
    a=ds[()]; return [v.decode('utf-8') if isinstance(v,bytes) else v for v in a]
def frame(g):
    out={}
    for key in g:
        x=g[key]
        if isinstance(x,h5py.Group) and 'categories' in x and 'codes' in x:
            cats=strings(x['categories']); codes=x['codes'][()]; out[key]=[cats[c] if c>=0 else None for c in codes]
        elif isinstance(x,h5py.Dataset): out[key]=strings(x) if x.ndim else []
    return pd.DataFrame(out)
audit=[]
for path in ROOT.glob('*.h5ad'):
    with h5py.File(path,'r') as f:
        obs=frame(f['obs']); var=frame(f['var']); name=path.name.split('_')[1]
        obs.to_csv(ROOT/(name+'_official_observations.csv'),index=False); var.to_csv(ROOT/(name+'_official_genes.csv'),index=False)
        x=f['X']; entry={'file':path.name,'rows':len(obs),'genes':len(var),'obs_columns':obs.columns.tolist(),'var_columns':var.columns.tolist(),'x_encoding':str(x.attrs.get('encoding-type')),'x_attributes':{k:str(v) for k,v in x.attrs.items()},'layers':list(f.get('layers',{}))}
        for c in ['Donor ID','Brain Region','library_prep','Supertype','Number of nuclei']:
            if c in obs: entry[c]={'unique':int(obs[c].nunique()),'examples':obs[c].dropna().astype(str).unique()[:15].tolist()}
        if isinstance(x,h5py.Group) and 'data' in x: entry['x_data_examples']=x['data'][:10].tolist()
        audit.append(entry)
(ROOT/'pseudobulk_structure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(audit,ensure_ascii=False,indent=2))
