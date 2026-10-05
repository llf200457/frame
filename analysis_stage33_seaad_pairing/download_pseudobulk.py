from pathlib import Path
import urllib.request,json,hashlib,time
ROOT=Path(__file__).resolve().parent
BASE='https://sea-ad-single-cell-profiling.s3.us-west-2.amazonaws.com/Multiregion_2026/pseudobulk_objects/'
log=[]
for cell,size in [('Astrocyte',169149269),('Immune',123541128)]:
    name=f'SEAAD_{cell}_HIP_MEC_LEC_ITG_MTG_FI_STG_DFC_AnG_V1C_RNAseq_final-nuclei_pseudobulked.2026-06-22.h5ad'; dest=ROOT/name; tmp=ROOT/(name+'.partial')
    if not dest.exists() or dest.stat().st_size!=size:
        for attempt in range(4):
            offset=tmp.stat().st_size if tmp.exists() else 0
            if offset==size: break
            try:
                req=urllib.request.Request(BASE+name,headers={'Range':f'bytes={offset}-','User-Agent':'public-research-download/1.0'})
                with urllib.request.urlopen(req,timeout=30) as response:
                    if offset and response.status!=206: raise RuntimeError('Resume request did not return partial content')
                    if response.status==206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'): raise RuntimeError('Unexpected byte range')
                    print(f'{cell}: response {response.status}, offset {offset}',flush=True)
                    with tmp.open('ab' if offset else 'wb') as f:
                        last=offset
                        while block:=response.read(1024*1024):
                            f.write(block)
                            if f.tell()-last>=32*1024*1024: print(f'{cell}: {f.tell()}/{size} bytes',flush=True); last=f.tell()
                if tmp.stat().st_size!=size: raise RuntimeError('Incomplete file')
                break
            except Exception as e:
                print(f'{cell}: attempt {attempt+1} {type(e).__name__}: {e}',flush=True)
                if attempt==3: raise
                time.sleep(2)
        assert tmp.stat().st_size==size
        with tmp.open('rb') as f: assert f.read(8)==b'\x89HDF\r\n\x1a\n'
        tmp.replace(dest)
    log.append({'celltype':cell,'url':BASE+name,'file':name,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    (ROOT/'download_manifest.json').write_text(json.dumps(log,indent=2),encoding='utf-8'); print('Verified '+cell,flush=True)
