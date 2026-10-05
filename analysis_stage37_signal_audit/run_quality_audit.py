from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import sys,json,hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'analysis_stage35_target_specific'))
from run_target_specific import prepare,fit_predict,weight
SRC=ROOT.parent/'analysis_stage33_seaad_pairing'
TARGET='percent AT8 positive area'

def load():
    keys=['Donor ID','brain region']
    p=pd.read_csv(SRC/'Astrocyte_paired_measured_pathology.csv'); o=pd.read_csv(SRC/'Immune_paired_measured_pathology.csv')
    p=p[p.qc_pass].merge(o.loc[o.qc_pass,keys+['row_index']],on=keys,suffixes=('_astro','_micro'),validate='one_to_one').sort_values(keys).reset_index(drop=True)
    meta=pd.read_csv(SRC/'SEA_AD_MTG_baseline_covariates.csv',index_col=0)
    cov=meta.loc[p['Donor ID']].reset_index(drop=True); features=[]; genes=[]; annotation=[]
    for cell,col in [('Astrocyte','row_index_astro'),('Immune','row_index_micro')]:
        ix=p[col].to_numpy(); raw=np.load(SRC/(cell+'_donor_region_counts.npz'))['counts'][ix]
        ann=pd.read_csv(SRC/(cell+'_official_genes.csv')); annotation.append(ann)
        genes.append(np.log2(1+raw/(raw.sum(axis=1,keepdims=True)/1e6)).astype(np.float32))
        qc=pd.read_csv(SRC/(cell+'_donor_region_index.csv')).set_index('row_index').loc[ix].reset_index(drop=True)
        assert (qc[keys].to_numpy()==p[keys].to_numpy()).all()
        z=pd.DataFrame({cell+'_log_nuclei':np.log1p(qc.nuclei),cell+'_log_UMI_per_nucleus':np.log1p(qc.total_UMI/qc.nuclei),cell+'_mito_fraction':raw[:,ann['index'].str.startswith('MT-').fillna(False).to_numpy()].sum(axis=1)/raw.sum(axis=1)})
        obs=pd.read_csv(SRC/(cell+'_official_observations.csv'),low_memory=False)
        if cell=='Immune': obs=obs[obs.Supertype.str.startswith('Micro-PVM',na=False)].copy()
        obs=obs.rename(columns={'Brain Region':'brain region'})
        n=pd.to_numeric(obs['Number of nuclei'],errors='raise'); obs['_n']=n
        totals=obs.groupby(keys)['_n'].sum()
        expected=totals.reindex(pd.MultiIndex.from_frame(p[keys])).to_numpy()
        assert np.allclose(expected,qc.nuclei)
        composition=obs.pivot_table(index=keys,columns='Supertype',values='_n',aggfunc='sum',fill_value=0)
        composition=composition.div(composition.sum(axis=1),axis=0).reindex(pd.MultiIndex.from_frame(p[keys])).iloc[:,1:]
        for name in composition: z[cell+'_fraction_'+name]=composition[name].to_numpy()
        for name in ['GEX_Median_genes_per_cell','GEX_Median_UMI_counts_per_cell','GEX_Reads_mapped_confidently_to_intronic_regions','GEX_Percent_duplicates']:
            v=pd.to_numeric(obs[name],errors='coerce'); valid=v.notna(); sub=obs.loc[valid,keys+['_n']].copy(); sub['_num']=v.loc[valid]*sub['_n']
            agg=sub.groupby(keys)[['_num','_n']].sum(); avg=(agg['_num']/agg['_n']).reindex(pd.MultiIndex.from_frame(p[keys])).to_numpy()
            if 'counts' in name or 'genes' in name: avg=np.log1p(avg)
            z[cell+'_'+name]=avg
        features.append(z)
    cov=pd.concat([cov,*features],axis=1)
    pd.concat([p[keys],cov],axis=1).to_csv(ROOT/'quality_covariates.csv',index=False)
    return p,cov,genes,annotation

def main():
    p,cov,genes,annotation=load(); donors=p['Donor ID'].to_numpy(); regions=p['brain region'].to_numpy(); y=np.log1p(p[TARGET].to_numpy(float)); ids=np.array(sorted(set(donors)))
    outputs=[]; logs=[]; innerlogs=[]
    for seed in [2026,2027]:
      for held in sorted(set(regions)):
        for fold,(_,tp) in enumerate(KFold(5,shuffle=True,random_state=seed).split(ids)):
            td=ids[tp]; tr=np.flatnonzero((regions!=held)&~np.isin(donors,td)); te=np.flatnonzero((regions==held)&np.isin(donors,td))
            assert not set(donors[tr])&set(donors[te]) and held not in regions[tr]
            trainids=np.array(sorted(set(donors[tr]))); groups=np.array_split(sorted(set(regions[tr])),3); inner=[]
            for j,(_,vp) in enumerate(KFold(3,shuffle=True,random_state=seed+fold).split(trainids)):
                vd=trainids[vp]; vr=groups[j]; it=tr[~np.isin(donors[tr],vd)&~np.isin(regions[tr],vr)]; iv=tr[np.isin(donors[tr],vd)&np.isin(regions[tr],vr)]
                assert len(iv)>0 and not set(donors[it])&set(donors[iv]) and not set(regions[it])&set(regions[iv])
                inner.append((it,iv)); innerlogs.append(dict(seed=seed,held=held,fold=fold,inner=j,train_donors=sorted(set(donors[it])),valid_donors=sorted(set(donors[iv])),train_regions=sorted(set(regions[it])),valid_regions=sorted(set(regions[iv]))))
            for name,kind in [('quality_only','metadata'),('quality_gene','pooled')]:
                grid=[10.,100.,1000.] if kind=='metadata' else [.01,.1,1.]
                cached=[(it,iv,*prepare(cov.to_numpy(),genes,y,donors,regions,it,iv,kind)[:2]) for it,iv in inner]; scores=[]
                for alpha in grid:
                    scores.append(float(np.mean([np.average(abs(fit_predict(a,b,y[it],donors[it],alpha,kind)[0]-y[iv]),weights=weight(donors[iv])) for it,iv,a,b in cached])))
                alpha=grid[int(np.argmin(scores))]; a,b,selected=prepare(cov.to_numpy(),genes,y,donors,regions,tr,te,kind); pred,model=fit_predict(a,b,y[tr],donors[tr],alpha,kind)
                logs.append(dict(seed=seed,held=held,fold=fold,model=name,alpha=alpha,inner_mae=scores,train_donors=sorted(set(donors[tr])),test_donors=sorted(set(donors[te])),train_regions=sorted(set(regions[tr])),selected_genes=[dict(cell=c,ids=ann.gene_ids.iloc[ii].tolist(),symbols=ann['index'].iloc[ii].tolist()) for c,ann,ii in zip(['Astrocyte','Micro-PVM'],annotation,selected)]))
                for j,row in enumerate(te): outputs.append(dict(donor_id=donors[row],region=held,fold=fold,seed=seed,model=name,target=TARGET,observed=y[row],predicted=pred[j]))
            pd.DataFrame(outputs).to_csv(ROOT/'predictions.csv',index=False); (ROOT/'training_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
            print(f'seed {seed} {held} fold {fold} complete',flush=True)
    (ROOT/'inner_splits.json').write_text(json.dumps(innerlogs,indent=2),encoding='utf-8')
    (ROOT/'run_manifest.json').write_text(json.dumps(dict(people=len(ids),records=len(p),fits=len(logs),predictions=len(outputs),covariate_columns=cov.columns.tolist(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),source_engine_sha256=hashlib.sha256((ROOT.parent/'analysis_stage35_target_specific/run_target_specific.py').read_bytes()).hexdigest()),indent=2),encoding='utf-8')
if __name__=='__main__': main()
