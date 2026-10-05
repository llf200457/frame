"""Download untouched GEO archives for external-cohort eligibility review.

Run from a network-enabled environment. This script never overwrites a nonempty
archive and records SHA256 after transfer. GEO source files remain unmodified.
"""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import shutil

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE if (BASE/'modules').is_dir() else BASE.parents[1]
RAW = ROOT / "raw"
FILES = {
    "GSE140829": {
        "GSE140829_final_normalized_data.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE140nnn/GSE140829/suppl/GSE140829_final_normalized_data.txt.gz",
        "GSE140829_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE140nnn/GSE140829/matrix/GSE140829_series_matrix.txt.gz",
    },
    "GSE97760": {
        "GSE97760_loess.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE97nnn/GSE97760/suppl/GSE97760_loess.txt.gz",
        "GSE97760_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE97nnn/GSE97760/matrix/GSE97760_series_matrix.txt.gz",
    },
}

manifest = []
for accession, entries in FILES.items():
    folder = RAW / accession
    folder.mkdir(parents=True, exist_ok=True)
    for filename, url in entries.items():
        path = folder / filename
        if not path.is_file() or path.stat().st_size == 0:
            temporary = path.with_suffix(path.suffix + ".part")
            with urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"accession": accession, "file": filename,
                         "url": url, "bytes": path.stat().st_size,
                         "sha256": digest})
        print(accession, filename, path.stat().st_size)

out = ROOT / "analysis_stage26" / "download_manifest.json"
out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
