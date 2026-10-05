"""Inspect or download explicitly selected public inputs, checking historic hashes."""
import argparse,csv,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--list',action='store_true')
    ap.add_argument('--dataset',help='Download rows for exactly this dataset ID; explicit choice required')
    a=ap.parse_args()
    with (ROOT/'docs/data_sources.csv').open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    downloads=[r for r in records if r['kind']=='download']
    if a.list or not a.dataset:
        for r in downloads:print(r['dataset'],r['historical_bytes'],r['destination'])
        return
    selected=[r for r in downloads if r['dataset']==a.dataset]
    if not selected:ap.error('No downloadable input registered for this dataset; see catalogue and historical scripts')
    for r in selected:
        dest=(ROOT/r['destination']).resolve();assert dest.is_relative_to(ROOT.resolve())
        if dest.exists():
            if hashlib.sha256(dest.read_bytes()).hexdigest()!=r['historical_sha256']:raise RuntimeError('Existing input checksum differs: '+str(dest))
            print('Already verified:',r['destination']);continue
        dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.name+'.partial')
        h=hashlib.sha256();size=0
        with urllib.request.urlopen(r['url'],timeout=60) as response,temp.open('wb') as f:
            while chunk:=response.read(1024*1024):f.write(chunk);h.update(chunk);size+=len(chunk)
        if size!=int(r['historical_bytes']) or h.hexdigest()!=r['historical_sha256']:raise RuntimeError('Downloaded input differs; partial retained for inspection: '+str(temp))
        temp.replace(dest);print('Verified:',r['destination'])
if __name__=='__main__':main()
