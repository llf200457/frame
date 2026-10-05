import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
RAW = OUT / 'raw'
ids = pd.read_csv(RAW / 'SRP429744_ena_runs.tsv', sep='\t').sample_accession.tolist()
items = [('SRP429744_biosamples_' + str(i // 35 + 1) + '.xml',
          'https://www.ebi.ac.uk/ena/browser/api/xml/' + ','.join(ids[i:i + 35])) for i in range(0, len(ids), 35)]
items += [(pmc + '_full.xml', 'https://www.ebi.ac.uk/europepmc/webservices/rest/' + pmc + '/fullTextXML')
          for pmc in ['PMC10872590', 'PMC10509252', 'PMC11077736']]


def fetch(item):
    name, url = item
    record = dict(name=name, url=url)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'PublicResearchAudit/1.0'}), timeout=25) as response:
            content = response.read(25 * 1024 * 1024 + 1)
            assert len(content) <= 25 * 1024 * 1024
            record.update(http_status=response.status, final_url=response.url)
        (RAW / name).write_bytes(content)
        record.update(ok=True, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    except Exception as exc:
        record.update(ok=False, error=repr(exc))
    print(name, record.get('ok'), record.get('bytes', record.get('error')), flush=True)
    return record


with ThreadPoolExecutor(max_workers=3) as pool:
    records = [f.result() for f in as_completed([pool.submit(fetch, item) for item in items])]
(OUT / 'batch_download_manifest.json').write_text(json.dumps(dict(no_prediction_scoring=True,
    no_raw_sequence_download=True, records=records), indent=2), encoding='utf-8')
