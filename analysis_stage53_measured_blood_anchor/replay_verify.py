"""One complete replay, compare every numeric analysis CSV, preserve logs."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parent
files=sorted(p for p in OUT.glob('*.csv'))
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
proc=subprocess.run([sys.executable,str(OUT/'run_measured_anchor_pilot.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
(OUT/'pilot_replay_execution.log').write_text(proc.stdout,encoding='utf-8')
assert proc.returncode==0,proc.stdout
after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
checks={k:before[k]==after[k] for k in before}
assert all(checks.values()),checks
result=dict(numeric_csv_files_checked=len(checks),all_byte_identical=True,files=checks)
(OUT/'replay_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
