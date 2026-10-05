from pathlib import Path
import gzip,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    annotations=[]; samples=[]; current=None; mode=None; columns=None
    with gzip.open(ROOT/'GSE106241_family.soft.gz','rt',encoding='utf-8',errors='replace') as f:
        for line in f:
            line=line.rstrip('\n\r')
            if line=='!platform_table_begin': mode='platform'; columns=None; continue
            if line=='!platform_table_end': mode=None; continue
            if line.startswith('^SAMPLE = '): current={'gsm':line.partition(' = ')[2],'values':{}}; samples.append(current)
            if line=='!sample_table_begin': mode='sample'; columns=None; continue
            if line=='!sample_table_end': mode=None; continue
            if mode:
                cells=line.split('\t')
                if columns is None: columns=cells; continue
                if mode=='platform': annotations.append(dict(zip(columns,cells)))
                else: current['values'][cells[0]]=float(cells[1]) if cells[1] not in ['null','NA',''] else np.nan
            elif current is not None and line.startswith('!Sample_characteristics_ch1 = '):
                key,_,value=line.partition(' = ')[2].partition(':'); current[key.strip().lower()]=value.strip()
    anno=pd.DataFrame(annotations).set_index('ID')
    expression=pd.DataFrame({s['gsm']:s['values'] for s in samples})
    meta=pd.DataFrame([{k:v for k,v in s.items() if k!='values'} for s in samples]).set_index('gsm')
    for c in meta.columns:
        if c not in ['tissue','brain region']: meta[c]=pd.to_numeric(meta[c],errors='coerce')
    assert expression.shape[1]==60 and expression.columns.tolist()==meta.index.tolist()
    symbols=anno.GeneSymbol.reindex(expression.index).fillna('').astype(str).str.strip()
    valid=symbols.ne('')&~symbols.str.contains(r'[;/]',regex=True)&np.isfinite(expression).all(axis=1)
    assert (expression.loc[valid]>=0).all().all()
    gene=np.log2(expression.loc[valid]+1).groupby(symbols[valid]).mean()
    # Multiple/exonic probe signals collapsed operationally; not equivalent to absolute RNA-seq counts.
    gene.to_pickle(ROOT/'GSE106241_gene_logsignal.pkl'); meta.to_csv(ROOT/'GSE106241_marker_metadata.csv')
    out={'samples':len(meta),'available_probe_rows':len(expression),'unambiguous_gene_logsignal_rows':len(gene),'targets':{k:int(meta[k].notna().sum()) for k in ['braak stage','amyloid-beta 42 levels','beta secretase activity','alpha secretase activity','gamma secretase activity']},'source_processing':'Depositor normexp background correction, cohort-wide quantile normalization and transcript/exon probe filtering; not independent raw-array normalization. Probe collapse is operational.'}
    (ROOT/'brain_endpoint_feasibility.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
