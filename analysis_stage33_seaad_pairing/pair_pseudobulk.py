from pathlib import Path
import json
import h5py
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent; PREV=ROOT.parent/'analysis_stage32_marker_pivot'
qnp=pd.read_csv(PREV/'SEA_AD_QNP_multi_region.csv'); qnp=qnp[qnp['analysis region'].eq('Grey matter')].copy()
assert not qnp.duplicated(['Donor ID','brain region']).any()
reports=[]
for cell in ['Astrocyte','Immune']:
    obs=pd.read_csv(ROOT/(cell+'_official_observations.csv')); var=pd.read_csv(ROOT/(cell+'_official_genes.csv'))
    keep=obs.Supertype.astype(str).str.startswith('Micro-PVM') if cell=='Immune' else np.ones(len(obs),bool)
    use=obs.loc[keep].copy(); keys=sorted(set(zip(use['Donor ID'],use['Brain Region']))); lookup={k:i for i,k in enumerate(keys)}
    counts=np.zeros((len(keys),len(var)),np.float64); nn=np.zeros(len(keys),np.int64)
    path=next(ROOT.glob('SEAAD_'+cell+'_*.h5ad'))
    with h5py.File(path,'r') as f:
        assert f['X'].shape==(len(obs),len(var))
        for start in range(0,len(obs),128):
            stop=min(start+128,len(obs)); mask=np.asarray(keep)[start:stop]
            block=f['X'][start:stop][mask]
            assert np.isfinite(block).all() and (block>=0).all() and np.equal(block,np.rint(block)).all()
            tmp=obs.iloc[start:stop].loc[mask]
            ids=np.array([lookup[k] for k in zip(tmp['Donor ID'],tmp['Brain Region'])])
            np.add.at(counts,ids,block); np.add.at(nn,ids,pd.to_numeric(tmp['Number of nuclei']).to_numpy(np.int64))
    index=pd.DataFrame(keys,columns=['Donor ID','brain region']); index['nuclei']=nn; index['total_UMI']=counts.sum(axis=1); index['row_index']=np.arange(len(index)); index['qc_pass']=(nn>=20)&(index.total_UMI>0)
    assert var.gene_ids.is_unique
    np.savez_compressed(ROOT/(cell+'_donor_region_counts.npz'),counts=counts,gene_ids=var.gene_ids.astype(str).to_numpy(),gene_symbols=var['index'].astype(str).to_numpy())
    index.to_csv(ROOT/(cell+'_donor_region_index.csv'),index=False)
    paired=index.merge(qnp,on=['Donor ID','brain region'],how='inner',validate='one_to_one'); paired.to_csv(ROOT/(cell+'_paired_measured_pathology.csv'),index=False)
    reports.append({'celltype':cell,'source_rows':len(obs),'included_source_rows':len(use),'donor_region_rows':len(index),'unique_donors':index['Donor ID'].nunique(),'genes':len(var),'qc_pass_rows':int(index.qc_pass.sum()),'pathology_paired_rows':len(paired),'paired_qc_by_region':paired[paired.qc_pass].groupby('brain region')['Donor ID'].nunique().to_dict(),'not_in_pathology_regions':sorted(set(index['brain region'])-set(qnp['brain region']))})
(ROOT/'pairing_audit.json').write_text(json.dumps(reports,indent=2),encoding='utf-8'); print(json.dumps(reports,indent=2))
