from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import gzip,tarfile,json,hashlib,argparse
import numpy as np
import pandas as pd
from scipy.io import mmread
import joblib
ROOT=Path(__file__).resolve().parent
PANELS={'Astrocyte':['AQP4','GFAP','ALDH1L1','SLC1A3','SLC1A2','GJA1'],'Micro-PVM':['P2RY12','TMEM119','CSF1R','C1QA','C1QB','TYROBP','PTPRC','AIF1','LAPTM5'],'Neuron':['SNAP25','RBFOX3','SYT1','SLC17A7','GAD1','GAD2'],'Oligo':['MBP','MOG','PLP1','MAG','MOBP'],'OPC':['PDGFRA','CSPG4','VCAN','OLIG1','OLIG2'],'Vascular':['CLDN5','FLT1','KDR','PECAM1','RGS5','PDGFRB']}
def read_archive(path):
 with tarfile.open(path,'r:gz') as tar:
  member={m.name.rsplit('/',1)[-1]:m for m in tar.getmembers() if m.isfile()}
  features=pd.read_csv(gzip.GzipFile(fileobj=tar.extractfile(member['features.tsv.gz'])),sep='\t',header=None,names=['gene_id','symbol','feature_type'])
  barcodes=pd.read_csv(gzip.GzipFile(fileobj=tar.extractfile(member['barcodes.tsv.gz'])),sep='\t',header=None)[0].to_numpy()
  features['original_gene_id']=features.gene_id
  features['gene_id']=features.gene_id.str.strip().str.replace(r'\.\d+(?=_PAR_Y$|$)','',regex=True)
  features['symbol']=features.symbol.str.strip()
  matrix=mmread(gzip.GzipFile(fileobj=tar.extractfile(member['matrix.mtx.gz'])),spmatrix=True).tocsc();matrix.eliminate_zeros()
 assert matrix.shape==(len(features),len(barcodes));assert not features.gene_id.duplicated().any();return features,barcodes,matrix
def process(meta):
 path=ROOT/'raw'/meta.filename;features,barcodes,mat=read_archive(path);n_genes=np.diff(mat.indptr);umi=np.asarray(mat.sum(axis=0)).ravel();mt=features.symbol.str.startswith('MT-').to_numpy();mt_fraction=np.asarray(mat[mt].sum(axis=0)).ravel()/np.maximum(umi,1);qc=(n_genes>=200)&(n_genes<=6000)&(umi<=25000)&(mt_fraction<=.05)
 scores=[];detected=[];coverage=[]
 for cell,genes in PANELS.items():
  ii=[]
  for sym in genes:
   ix=np.flatnonzero(features.symbol.eq(sym));assert len(ix)==1,(sym,len(ix));ii.append(ix[0])
  raw=mat[ii].toarray();z=np.log1p(raw*10000/np.maximum(umi,1));scores.append(z.mean(axis=0));detected.append((raw>0).sum(axis=0));coverage.extend(dict(gsm=meta.gsm,cell=cell,symbol=sym,detected_n=int((raw[i]>0).sum())) for i,sym in enumerate(genes))
 scores=np.array(scores).T;detected=np.array(detected).T;win=scores.argmax(axis=1);sorted_scores=np.sort(scores,axis=1);margin=sorted_scores[:,-1]-sorted_scores[:,-2];n_detect=detected[np.arange(len(win)),win];cells=np.array(list(PANELS));qc_rows=pd.DataFrame({'barcode':barcodes,'n_genes':n_genes,'umi':umi,'mt_fraction':mt_fraction,'qc_pass':qc,'marker_label':cells[win],'marker_margin':margin,'winning_panel_genes_detected':n_detect})
 for i,c in enumerate(cells):qc_rows['score_'+c]=scores[:,i]
 (ROOT/'nucleus_qc').mkdir(exist_ok=True);qc_rows.to_csv(ROOT/'nucleus_qc'/(meta.gsm+'.csv.gz'),index=False)
 models={target:joblib.load(ROOT/('frozen_'+target+'.joblib')) for target in ['Tau_panel','Tau_DFC']};rows=[];counts=[];summaries=[];pseudobulk=[]
 for threshold in [.25,.50]:
  selected=qc&(n_detect>=2)&(margin>=threshold);expressions={}
  for i,c in enumerate(cells):
   keep=selected&(win==i);counts.append(dict(gsm=meta.gsm,donor_id=meta.donor_id,region=meta['brain region (somatosensory or entorhinal cortex)'],marker_margin=threshold,cell=c,nuclei=int(keep.sum()),total_input_nuclei=len(win),qc_pass_nuclei=int(qc.sum()),unclassified_qc_nuclei=int((qc&~selected).sum())))
   if c not in ['Astrocyte','Micro-PVM']:continue
   if keep.sum()<20:raise RuntimeError(f'{meta.gsm} {c} margin{threshold}: too few nuclei {keep.sum()}')
   raw=np.asarray(mat[:,keep].sum(axis=1)).ravel();log=np.log2(1+raw*1e6/raw.sum());expressions[c]=dict(zip(features.gene_id,log))
   pseudobulk.append(dict(cell=c,marker_margin=threshold,counts=raw))
   for scorecell in cells:summaries.append(dict(gsm=meta.gsm,marker_margin=threshold,assigned_cell=c,marker_panel=scorecell,mean_panel_score=float(scores[keep,cells.tolist().index(scorecell)].mean())))
  for target,bundle in models.items():
   ids=[];symbols=[];values=[]
   for sel in bundle['selected_genes']:
    cell=sel['cell'];missing=set(sel['gene_ids'])-set(expressions[cell]);
    if missing:raise RuntimeError(f'Missing required genes {sorted(missing)}')
    ids+=sel['gene_ids'];symbols+=sel['symbols'];values.extend(expressions[cell][g] for g in sel['gene_ids'])
   values=np.asarray(values);scaled=(values-bundle['joint_scaler'].mean_[30:])/bundle['joint_scaler'].scale_[30:];contributions=scaled*bundle['model'].coef_[30:]*bundle['target_sd'];nonmt=~pd.Series(symbols).str.startswith('MT-').to_numpy()
   for component,keep in [('all',np.ones(20,dtype=bool)),('no_MT',nonmt)]:rows.append(dict(gsm=meta.gsm,donor_id=meta.donor_id,region=meta['brain region (somatosensory or entorhinal cortex)'],target=target,component=component,marker_margin=threshold,frozen_RNA_score=float(contributions[keep].sum()),ptau_percent=float(meta['ptau immunostaining (%ptau positive cells)']),amyloid_percent=float(meta['amyloid immunostaining (%area stained)'])))
 (ROOT/'pseudobulk').mkdir(exist_ok=True);np.savez_compressed(ROOT/'pseudobulk'/(meta.gsm+'.npz'),gene_ids=features.gene_id.to_numpy(dtype=str),symbols=features.symbol.to_numpy(dtype=str),**{f'{p["cell"]}_{p["marker_margin"]}':p['counts'] for p in pseudobulk})
 return rows,counts,summaries,coverage
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--gsm');args=parser.parse_args();meta=pd.read_csv(ROOT/'external_metadata.csv');rows=[];counts=[];summaries=[];coverage=[]
 if args.gsm:meta=meta[meta.gsm==args.gsm]
 for _,r in meta.iterrows():
  a,b,c,d=process(r);rows+=a;counts+=b;summaries+=c;coverage+=d;print(r.gsm,'done',len(a),'scores',flush=True)
 for name,data in [('external_scores',rows),('cell_qc_counts',counts),('annotation_marker_summary',summaries),('annotation_gene_coverage',coverage)]:pd.DataFrame(data).to_csv(ROOT/(name+('_smoke' if args.gsm else '')+'.csv'),index=False)
 (ROOT/('processing_manifest_smoke.json' if args.gsm else 'processing_manifest.json')).write_text(json.dumps({'samples':len(meta),'independent_donors':meta.donor_id.nunique(),'input':'raw sum per operationally marker-annotated class; log2(1+CPM), full gene denominator','frozen_RNA_component_not_full_prediction':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
if __name__=='__main__':main()
