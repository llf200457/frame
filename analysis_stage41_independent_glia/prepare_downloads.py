from pathlib import Path
import gzip,json,shutil,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parent;SRC=ROOT.parent/'analysis_stage40_multiview';RAW=ROOT/'raw';RAW.mkdir(exist_ok=True)
soft=SRC/'GSE160936_family.soft.gz';text=gzip.open(soft,'rt',encoding='utf-8').read();rows=[]
for block in text.split('^SAMPLE = ')[1:]:
 lines=block.splitlines();row={'gsm':lines[0].strip()}
 for line in lines[1:]:
  if line.startswith('!Sample_characteristics_ch1 = '):
   key,val=line.split(' = ',1)[1].split(': ',1);row[key.lower()]=val
  elif line.startswith('!Sample_title = '):row['title']=line.split(' = ',1)[1];row['donor_id']=row['title'].rsplit(' ',1)[0]
  elif line.startswith('!Sample_supplementary_file_1 = '):row['url']=line.split(' = ',1)[1].replace('ftp://','https://');row['filename']=row['url'].rsplit('/',1)[-1]
 rows.append(row)
d=pd.DataFrame(rows);assert len(d)==24 and d.donor_id.nunique()==12
d.to_csv(ROOT/'external_metadata.csv',index=False);shutil.copy2(soft,RAW/soft.name)
src=SRC/'GSM4886750_IGF112640_filtered_feature_bc_matrix.tar.gz'
if src.exists():shutil.copy2(src,RAW/src.name)
shutil.copy2(ROOT.parent/'analysis_stage39_tau_panel/frozen_models/Tau_panel.joblib',ROOT/'frozen_Tau_panel.joblib');shutil.copy2(ROOT.parent/'analysis_stage39_tau_panel/frozen_models/Tau_DFC.joblib',ROOT/'frozen_Tau_DFC.joblib')
(ROOT/'freeze_manifest.json').write_text(json.dumps({'freeze_before_external_expression_processing':True,'pathology_units':'GEO %pTau-positive cells; not SEA-AD AT8 area %','n_donors':12,'n_samples':24,'source_publication':'https://pmc.ncbi.nlm.nih.gov/articles/PMC8732962/','protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),'model_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.joblib')}},indent=2),encoding='utf-8')
print(d[['gsm','donor_id','brain region (somatosensory or entorhinal cortex)','ptau immunostaining (%ptau positive cells)']].to_string(index=False))
