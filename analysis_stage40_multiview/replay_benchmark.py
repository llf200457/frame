from pathlib import Path
import hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
files=['predictions.csv','training_log.json','smoke_checks.json']
backup=ROOT/'replay_before';backup.mkdir(exist_ok=True)
before={}
for f in files:
 data=(ROOT/f).read_bytes();(backup/f).write_bytes(data);before[f]=hashlib.sha256(data).hexdigest()
subprocess.run([sys.executable,'-X','utf8',str(ROOT/'run_benchmark.py')],check=True)
after={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
audit={'before':before,'after':after,'byte_identical':{f:before[f]==after[f] for f in files},'scope':'405 primary benchmark outer fits replayed; follow-up and external analyses not replayed in this check. Runtime manifest intentionally excluded.'}
(ROOT/'replay_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
assert all(audit['byte_identical'].values());print('PRIMARY BENCHMARK REPLAY: all three audited files byte-identical')
