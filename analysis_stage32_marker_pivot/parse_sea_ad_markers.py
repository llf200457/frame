from pathlib import Path
import zipfile,xml.etree.ElementTree as ET,json,hashlib,re
import pandas as pd
ROOT=Path(__file__).resolve().parent
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def sheets(path):
    out={}
    with zipfile.ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            shared=[''.join(n.text or '' for n in el.iter() if n.tag.endswith('}t')) for el in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels={x.attrib['Id']:x.attrib['Target'].lstrip('/') for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets',NS):
            path=rels[sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']]; path=path if path.startswith('xl/') else 'xl/'+path
            rows=[]
            for row in ET.fromstring(z.read(path)).find('s:sheetData',NS):
                vals={}
                for c in row:
                    letters=re.match('[A-Z]+',c.attrib['r']).group(); idx=0
                    for char in letters: idx=idx*26+ord(char)-64
                    el=c.find('s:v',NS); value=el.text if el is not None else ''.join(n.text or '' for n in c.iter() if n.tag.endswith('}t'))
                    if c.attrib.get('t')=='s': value=shared[int(value)]
                    vals[idx-1]=value
                rows.append([vals.get(i,'') for i in range(max(vals,default=-1)+1)])
            out[sheet.attrib['name']]=rows
    return out

def frame(rows,start=0):
    header=rows[start]; width=len(header)
    return pd.DataFrame([(r+['']*width)[:width] for r in rows[start+1:]],columns=header).replace('',pd.NA)

meta=frame(next(iter(sheets(ROOT/'SEA_AD_donor_metadata.xlsx').values())))
meta=meta[meta['Donor ID'].notna()]; meta.to_csv(ROOT/'SEA_AD_donor_metadata.csv',index=False)
raw=next(iter(sheets(ROOT/'SEA_AD_Luminex_MTG.xlsx').values()))
raw[1]=['Donor ID']+[f'{buffer}_{marker}' for buffer in ['RIPA','GuHCl'] for marker in ['ABeta40_pg_per_ug','ABeta42_pg_per_ug','tTAU_pg_per_ug','pTAU_pg_per_ug']]
lum=frame(raw,1); lum=lum[lum['Donor ID'].notna()]; lum.to_csv(ROOT/'SEA_AD_Luminex_MTG.csv',index=False)
qnp=frame(sheets(ROOT/'SEA_AD_quantitative_neuropathology_063026.xlsx')['Travaglini 2026 (multi-region)'])
qnp=qnp[qnp['Donor ID'].notna()]; qnp.to_csv(ROOT/'SEA_AD_QNP_multi_region.csv',index=False)
targets=['percent 6e10 positive area','percent AT8 positive area','percent GFAP positive area','number of Iba1 positive cells per area']
out={'donor_metadata':{'rows':len(meta),'unique_donors':meta['Donor ID'].nunique()},'Luminex':{'rows':len(lum),'unique_donors':lum['Donor ID'].nunique(),'overlap_metadata':len(set(lum['Donor ID'])&set(meta['Donor ID'])),'numeric_target_counts':{c:int(pd.to_numeric(lum[c],errors='coerce').notna().sum()) for c in lum.columns[1:]}},'QNP':{'rows':len(qnp),'unique_donors':qnp['Donor ID'].nunique(),'regions':qnp['brain region'].value_counts().to_dict(),'duplicate_donor_region':int(qnp.duplicated(['Donor ID','brain region','analysis region']).sum()),'target_availability_by_region':{region:{c:int(pd.to_numeric(sub[c],errors='coerce').notna().sum()) for c in targets} for region,sub in qnp.groupby('brain region')}},'warning':'Multiple regions share donors. Split by Donor ID; distinguish soluble/insoluble buffer-specific biochemical measures from image stain burdens. No paired transcriptome has yet been downloaded or modeled.','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('SEA_AD*.xlsx')}}
(ROOT/'SEA_AD_marker_feasibility.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
