"""Fetch public GEO files for eligibility review; records source checksums."""
from pathlib import Path
from hashlib import sha256
import json
import urllib.request

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE if (BASE / 'modules').is_dir() else BASE.parents[1]
OUT = ROOT / 'raw' / 'GSE249477'
OUT.mkdir(parents=True, exist_ok=True)
base = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE249nnn/GSE249477/'
files = {
    'GSE249477_raw_count_normalize_04-10-2025.csv.gz': base + 'suppl/GSE249477_raw_count_normalize_04-10-2025.csv.gz',
    'GSE249477_series_matrix.txt.gz': base + 'matrix/GSE249477_series_matrix.txt.gz',
}
manifest = {}
for name, url in files.items():
    dest = OUT / name
    if not dest.exists():
        temp = dest.with_suffix(dest.suffix + '.part')
        req = urllib.request.Request(url, headers={'User-Agent': 'research-data-audit/1.0'})
        with urllib.request.urlopen(req, timeout=90) as response, temp.open('wb') as fh:
            while chunk := response.read(1024 * 1024):
                fh.write(chunk)
        temp.replace(dest)
    h = sha256()
    with dest.open('rb') as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    manifest[name] = {'url': url, 'size_bytes': dest.stat().st_size, 'sha256': h.hexdigest()}
(ROOT / 'analysis_stage27' / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
