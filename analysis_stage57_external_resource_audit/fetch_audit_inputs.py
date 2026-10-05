"""Small anonymous public inputs only; no FASTQ download or model scoring."""
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent
RAW = OUT / 'raw'
RAW.mkdir(exist_ok=True)

items = [
    ('MCPcounter_commit.json', 'https://api.github.com/repos/ebecht/MCPcounter/commits/master'),
    ('SRP429744_ena_runs.tsv', 'https://www.ebi.ac.uk/ena/portal/api/filereport?' + urllib.parse.urlencode(dict(
        accession='SRP429744', result='read_run', fields='run_accession,study_accession,secondary_study_accession,sample_accession,secondary_sample_accession,sample_alias,experiment_accession,scientific_name,library_source,library_strategy,library_layout,fastq_ftp,fastq_bytes', format='tsv'))),
    ('SRP429744_ena_study.xml', 'https://www.ebi.ac.uk/ena/browser/api/xml/SRP429744'),
    ('SRP429744_ena_analysis.tsv', 'https://www.ebi.ac.uk/ena/portal/api/filereport?' + urllib.parse.urlencode(dict(
        accession='SRP429744', result='analysis', fields='analysis_accession,study_accession,analysis_title,submitted_ftp', format='tsv'))),
]


def fetch(item):
    name, url = item
    record = dict(name=name, url=url, requested_utc=datetime.now(timezone.utc).isoformat())
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PublicResearchAudit/1.0', 'Accept': '*/*'})
        with urllib.request.urlopen(req, timeout=25) as response:
            content = response.read(25 * 1024 * 1024 + 1)
            assert len(content) <= 25 * 1024 * 1024, 'Unexpectedly large audit input'
            record.update(http_status=response.status, final_url=response.url, content_type=response.headers.get('Content-Type'))
        (RAW / name).write_bytes(content)
        record.update(ok=True, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    except Exception as exc:
        record.update(ok=False, error=repr(exc))
    print(name, record.get('ok'), record.get('bytes', record.get('error')), flush=True)
    return record


with ThreadPoolExecutor(max_workers=3) as pool:
    records = [f.result() for f in as_completed([pool.submit(fetch, item) for item in items])]
commit_record = next(v for v in records if v['name'] == 'MCPcounter_commit.json')
ref = json.loads((RAW / 'MCPcounter_commit.json').read_text(encoding='utf-8'))['sha'] if commit_record['ok'] else 'master'
for name, path in [('MCPcounter.R', 'Source/R/MCPcounter.R'), ('MCPcounter_DESCRIPTION', 'Source/DESCRIPTION'),
                   ('MCPcounter_genes.txt', 'Signatures/genes.txt')]:
    records.append(fetch((name, f'https://raw.githubusercontent.com/ebecht/MCPcounter/{ref}/{path}')))
(OUT / 'download_manifest.json').write_text(json.dumps(dict(date='2026-10-03',
    mcp_upstream_ref=ref, no_prediction_scoring=True, no_raw_sequence_download=True,
    records=records), indent=2), encoding='utf-8')
