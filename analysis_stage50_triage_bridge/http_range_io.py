"""Verified HTTP byte-range access for immutable Zenodo HDF5, using system TLS.

Fetched blocks are SHA-256 recorded. This does NOT claim whole-file MD5
verification; the original repository length/checksum are retained separately.
"""
import hashlib
import io
import json
import re
import subprocess
import time
from pathlib import Path


class ZenodoRangeReader(io.RawIOBase):
    def __init__(self, url, size, cache, prefix=None, block=4 * 1024**2):
        super().__init__()
        self.url, self.size = url, int(size)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.prefix = Path(prefix) if prefix else None
        self.prefix_size = self.prefix.stat().st_size if self.prefix and self.prefix.exists() else 0
        self.block = block
        self.pos = 0
        self.journal = self.cache / 'range_download_manifest.json'
        self.events = json.loads(self.journal.read_text(encoding='utf-8')) if self.journal.exists() else []

    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        destination = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if destination < 0:
            raise ValueError('Negative offset')
        self.pos = int(destination)
        return self.pos

    def _fetch(self, start):
        end = min(start + self.block, self.size) - 1
        target = self.cache / f'{start:012d}_{end:012d}.bin'
        if target.exists():
            if target.stat().st_size != end - start + 1:
                raise IOError('Cached block length mismatch')
            return target
        part = target.with_suffix('.part')
        header = target.with_suffix('.headers')
        for attempt in range(3):
            command = ['curl.exe', '-sS', '-L', '--connect-timeout', '20', '--max-time', '60',
                       '--max-filesize', str(end-start+1), '--range', f'{start}-{end}',
                       '--dump-header', str(header), '--output', str(part), self.url]
            result = subprocess.run(command, capture_output=True, text=True)
            headers = header.read_text(errors='replace') if header.exists() else ''
            ranges = re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)', headers, re.I)
            if result.returncode == 0 and ranges and tuple(map(int, ranges[-1])) == (start, end, self.size) and part.stat().st_size == end-start+1:
                part.replace(target)
                event = {'url': self.url, 'start': start, 'end': end, 'whole_file_bytes': self.size,
                         'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                         'retrieved_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         'content_range_verified': True, 'whole_file_md5_verified': False}
                self.events.append(event)
                self.journal.write_text(json.dumps(self.events, indent=2), encoding='utf-8')
                print(f'Cached HDF5 block {start}-{end}', flush=True)
                return target
            if attempt == 2:
                raise IOError(f'HTTP range failed: {start}-{end}; curl={result.returncode}; {result.stderr[:300]}; ranges={ranges[-1:]}')
            time.sleep(2)

    def read(self, size=-1):
        if size is None or size < 0: size = self.size - self.pos
        size = max(0, min(int(size), self.size-self.pos))
        result = bytearray()
        while size:
            if self.pos < self.prefix_size:
                take = min(size, self.prefix_size - self.pos)
                with self.prefix.open('rb') as stream:
                    stream.seek(self.pos)
                    data = stream.read(take)
            else:
                start = self.pos // self.block * self.block
                file = self._fetch(start)
                take = min(size, start+self.block-self.pos, self.size-self.pos)
                with file.open('rb') as stream:
                    stream.seek(self.pos-start)
                    data = stream.read(take)
            if len(data) != take: raise IOError('Short read')
            result.extend(data)
            self.pos += take
            size -= take
        return bytes(result)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)
