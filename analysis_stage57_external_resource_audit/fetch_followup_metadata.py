"""Fetch actual measured metadata and inspect published processed-data leads."""
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
RAW = OUT / 'raw'


def fetch(name, url):
    record = dict(name=name, url=url)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PublicResearchAudit/1.0'})
        with urllib.request.urlopen(req, timeout=25) as response:
            content = response.read(25 * 1024 * 1024 + 1)
            assert len(content) <= 25 * 1024 * 1024
            record.update(http_status=response.status, final_url=response.url)
        (RAW / name).write_bytes(content)
        record.update(ok=True, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    except Exception as exc:
        record.update(ok=False, error=repr(exc))
    print(name, record.get('ok'), record.get('bytes', record.get('error')), flush=True)
    return record


items = [
    ('SRP429744_example_sample.xml', 'https://www.ebi.ac.uk/ena/browser/api/xml/SAMN33948019'),
    ('SLE_E-GEOD-65391.sdrf.txt', 'https://raw.githubusercontent.com/greenelab/rheum-plier-data/master/sle-wb/arrayexpress/E-GEOD-65391/E-GEOD-65391.sdrf.txt'),
    ('RNA_normalization_2023.html', 'https://www.nature.com/articles/s41598-023-41443-4'),
    ('RNA_CBC_2024.html', 'https://link.springer.com/article/10.1186/s12863-024-01223-z'),
    ('MCPcounter_README.md', 'https://raw.githubusercontent.com/ebecht/MCPcounter/' + json.loads((OUT / 'download_manifest.json').read_text())['mcp_upstream_ref'] + '/README.md'),
]
with ThreadPoolExecutor(max_workers=3) as pool:
    records = [f.result() for f in as_completed([pool.submit(fetch, *item) for item in items])]
# NCBI requests sequential and slower than the documented unauthenticated limit.
q = urllib.parse.urlencode(dict(db='biosample', term='PRJNA949611', retmax=200, retmode='json'))
search = fetch('PRJNA949611_biosample_search.json', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?' + q)
records.append(search)
if search['ok']:
    ids = json.loads((RAW / search['name']).read_text())['esearchresult']['idlist']
    time.sleep(.6)
    if ids:
        records.append(fetch('PRJNA949611_biosamples.xml', 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?' +
                             urllib.parse.urlencode(dict(db='biosample', id=','.join(ids), retmode='xml'))))
(OUT / 'followup_download_manifest.json').write_text(json.dumps(dict(date='2026-10-03', no_prediction_scoring=True,
    no_raw_sequence_download=True, records=records), indent=2), encoding='utf-8')
