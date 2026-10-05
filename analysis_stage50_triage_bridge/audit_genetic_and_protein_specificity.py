"""Check public genetic tools and H3-normalization specificity; no causal MR."""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent

def bh(p):
    p=np.asarray(p,float);ix=np.argsort(p)
    qq=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty(len(p));result[ix]=np.minimum(qq,1)
    return result

candidates=pd.read_csv(ROOT/'prespecified_genetic_candidate_module_links.csv')
with h5py.File(DATA/'analysis_stage49_top_journal_outlook/raw/omar_small_batch.h5','r') as h:
    f=h['matrix/features']
    names=[v.decode() for v in f['name'][:]]
    ids=[v.decode().split('.')[0] for v in f['id'][:]]
mapping={g:sorted(set(i for n,i in zip(names,ids) if n==g)) for g in candidates.gene}
rows=[]
for tissue,filename in [('monocyte','blueprint_monocyte_permuted.tsv.gz'),('neutrophil','blueprint_neutrophil_permuted.tsv.gz')]:
    table=pd.read_csv(ROOT/'raw'/filename,sep='\t')
    table['q_eGene_all']=bh(table.p_beta)
    for gene in candidates.gene:
        valid_ids=mapping[gene]
        match=table.loc[table.molecular_trait_id.astype(str).str.split('.').str[0].isin(valid_ids)]
        assert len(match)<=1,(gene,tissue,len(match))
        row={'gene':gene,'module':candidates.loc[candidates.gene==gene,'module'].iloc[0],'tissue':tissue,
             'gene_id_mapping_unique':len(valid_ids)==1,'ensembl_id':valid_ids[0] if len(valid_ids)==1 else None,
             'gene_tested':len(match)==1,'genetic_instrument_fully_validated':False}
        if len(match):
            record=match.iloc[0]
            row.update({k:record[k] for k in ['variant','pvalue','beta','p_beta','q_eGene_all','n_variants']})
            row['candidate_screen_pass']=bool(record.pvalue<5e-8 and record.q_eGene_all<.05)
        else: row['candidate_screen_pass']=False
        rows.append(row)
pd.DataFrame(rows).to_csv(ROOT/'cis_eqtl_candidate_screen.csv',index=False)

obs=pd.read_csv(ROOT/'microglia_obs.csv',index_col=0,dtype={'SampleID':str})
donors=pd.read_csv(ROOT/'paired_donor_protein_clinical_wb.csv',dtype={'donor':str}).set_index('donor')
eligible=obs.SampleID.isin(donors.index)&obs.total_counts_NFKB.notna()&(obs.total_counts_H3>1)
protein={}
for name,column in [('NFkB','total_counts_NFKB'),('TDP43','total_counts_TDP'),('beta_catenin','total_counts_BCATENIN')]:
    vals=np.log1p((obs.loc[eligible,column]+1)/obs.loc[eligible,'total_counts_H3'])
    protein[name]=vals.groupby(obs.loc[eligible,'SampleID']).mean().reindex(donors.index)
q=pd.DataFrame({'age':donors.Age,'male':donors.Sex.eq('M').astype(float),'RIN':donors.JoinedRIN,
                'log_RNA_UMI':np.log1p(donors.n_counts_mean),'log_H3_ADT':np.log1p(donors.H3_adt_mean)})
bundle=np.load(ROOT/'paired_microglia_pseudobulk.npz')
order=bundle['donors'].tolist();donors=donors.loc[order];q=q.loc[order]
genes=bundle['genes'].tolist();expression=bundle['log2cpm'];mu=expression.mean(axis=0);sd=expression.std(axis=0,ddof=1);sd[sd<1e-12]=1
z=(expression-mu)/sd
membership=pd.read_csv(DATA/'modules/frozen_pca_modules.csv')
lookup={g:i for i,g in enumerate(genes)}
results=[]
for module in ['M01','M07']:
    part=membership.loc[membership.module==module]
    score=sum(np.sign(row.loading_signed)*z[:,lookup[row.gene]] for row in part.itertuples() if row.gene in lookup)/len(part)
    for endpoint in ['NFkB','TDP43','beta_catenin','H3_ADT','NFkB_ADT']:
        target=protein[endpoint].loc[order].to_numpy() if endpoint in protein else np.log1p(donors.H3_adt_mean.to_numpy()) if endpoint=='H3_ADT' else np.log1p(donors.NFkB_adt_mean.to_numpy())
        for control in ['age_sex_RIN_RNA_diagnosis','plus_H3']:
            if endpoint=='H3_ADT' and control=='plus_H3':
                continue  # Never residualize an endpoint on itself.
            columns=['age','male','RIN','log_RNA_UMI']+(['log_H3_ADT'] if control=='plus_H3' else [])
            cov=q[columns].apply(pd.to_numeric,errors='coerce')
            cov=cov.fillna(cov.median())
            c=np.column_stack([np.ones(len(order)),cov.to_numpy(),pd.get_dummies(donors.DiseaseState,drop_first=True).to_numpy(float)])
            a,b=pd.Series(score).rank().to_numpy(),pd.Series(target).rank().to_numpy()
            ar=a-c@np.linalg.lstsq(c,a,rcond=None)[0];br=b-c@np.linalg.lstsq(c,b,rcond=None)[0]
            if np.std(ar)<1e-12 or np.std(br)<1e-12:
                r,p=None,None
            else:
                r=float(np.corrcoef(ar,br)[0,1]);df=len(order)-np.linalg.matrix_rank(c)-1
                p=float(2*t.sf(abs(r)*np.sqrt(df/max(1e-12,1-r*r)),df))
            results.append({'module':module,'endpoint':endpoint,'adjustment':control,'partial_rank_r':r,'p':p,'n':len(order),
                            'not_a_protein_prediction_validation':True})
result=pd.DataFrame(results);valid=result.p.notna();result.loc[valid,'q_bh']=bh(result.loc[valid,'p'])
result.to_csv(ROOT/'protein_anchor_specificity_controls.csv',index=False)
audit={'date':'2026-10-02','genetic_candidates':len(candidates),'candidate_tissue_tests':len(rows),
       'strong_nominal_and_gene_FDR_screen_pairs':int(sum(row['candidate_screen_pass'] for row in rows)),
       'genetic_exposures_are_gene_expression_not_nuclear_protein':True,
       'exact_eQTL_standard_errors_available_in_downloaded_permutation_summary':False,
       'instrument_F_not_computed_from_approximate_SE':True,'complete_locus_summary_and_AD_outcome_harmonization_pending':True,
       'MR_run':False,'colocalization_run':False,'AD_GWAS_specific_file':'GCST90027158_buildGRCh38.tsv.gz',
       'AD_GWAS_metadata_ancestry':'European','AD_GWAS_metadata_sample_size':487511,
       'eqtl_catalogue_REST_API_is_deprecated_verified':True,'virtual_knockout_run':False,
       'virtual_knockout_gate':'protein prediction and network validity required; direct classifier masking not a knockout',
       'protein_specificity_controls_n':len(result)}
(ROOT/'genetic_and_specificity_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(rows)[['gene','module','tissue','candidate_screen_pass']].to_string(index=False))
print(result.loc[(result.module=='M01')&(result.endpoint.isin(['NFkB','H3_ADT','NFkB_ADT']))].to_string(index=False))
