"""Inspect real paired annotations via bounded remote reads; do not train."""
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from http_range_io import ZenodoRangeReader

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw'
RECORD = json.loads((ROOT.parent/'analysis_stage49_top_journal_outlook/raw/omar_record.json').read_text(encoding='utf-8-sig'))
ENTRY = next(x for x in RECORD['files'] if x['key'] == 'Omar_et_al_MicrogliaData.h5ad')
URL = 'https://zenodo.org/records/14712707/files/Omar_et_al_MicrogliaData.h5ad?download=1'

def decode(values):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in values])

def read_column(obj):
    if isinstance(obj, h5py.Group) and obj.attrs.get('encoding-type') == 'categorical':
        codes = obj['codes'][:]
        categories = decode(obj['categories'][:])
        return np.array([categories[int(i)] if i >= 0 else None for i in codes], dtype=object)
    return decode(obj[:]) if obj.dtype.kind in 'OS' else obj[:]

def read_frame(group, selected=None):
    columns = decode(group.attrs['column-order'])
    if selected is not None:
        columns = [c for c in columns if str(c) in selected]
    index_key = group.attrs.get('_index', '_index')
    if isinstance(index_key, bytes): index_key = index_key.decode()
    index = decode(group[index_key][:])
    data = {}
    for col in columns:
        print('Reading annotation:', str(col), flush=True)
        data[str(col)] = read_column(group[str(col)])
    frame = pd.DataFrame(data, index=index)
    frame.index.name = 'barcode'
    return frame

with ZenodoRangeReader(URL, ENTRY['size'], RAW/'microglia_range_cache', RAW/'Omar_et_al_MicrogliaData.h5ad.part') as reader:
    with h5py.File(reader, 'r') as handle:
        print('Root groups:',list(handle), flush=True)
        layout = {}
        for path in ['X','raw/X','layers','layers/raw_counts']:
            if path not in handle: continue
            obj = handle[path]
            if isinstance(obj,h5py.Dataset):
                layout[path] = {'shape':list(obj.shape),'dtype':str(obj.dtype),'chunks':obj.chunks,'compression':obj.compression,'offset':obj.id.get_offset()}
            else:
                layout[path] = {'encoding':str(obj.attrs.get('encoding-type','')),'shape':np.asarray(obj.attrs.get('shape',[])).tolist(),'children':list(obj)}
                if 'data' in obj:
                    for child in ['data','indices','indptr']:
                        ds = obj[child]
                        layout[path+'/'+child] = {'shape':list(ds.shape),'dtype':str(ds.dtype),'chunks':ds.chunks,'compression':ds.compression,'offset':ds.id.get_offset()}
            print('Matrix layout:',path,layout[path],flush=True)
        (ROOT/'microglia_hdf5_layout.json').write_text(json.dumps(layout, indent=2), encoding='utf-8')
        obs_columns = [str(c) for c in decode(handle['obs'].attrs['column-order'])]
        (ROOT/'microglia_obs_column_inventory.json').write_text(json.dumps(obs_columns,indent=2),encoding='utf-8')
        selected = [c for c in obs_columns if any(k in c.lower() for k in ['sample','donor','nfkb','nfk','h3','tau','disease','date','batch','cluster','leiden','age','sex','gender','total_counts','n_genes','mito','pct_counts','region','rin','postmortem','erg','exclusion','prep_type','bank','n_counts','initial_size'])]
        obs = read_frame(handle['obs'],selected=selected)
        var = read_frame(handle['var'],selected=[])
        obs.to_csv(ROOT/'microglia_obs.csv', encoding='utf-8-sig')
        var.to_csv(ROOT/'microglia_var.csv', encoding='utf-8-sig')
        if 'raw' in handle and 'var' in handle['raw']:
            read_frame(handle['raw/var'],selected=[]).to_csv(ROOT/'microglia_raw_var.csv', encoding='utf-8-sig')
        audit = {'source_url': URL, 'repository_file_bytes': ENTRY['size'], 'repository_md5': ENTRY['checksum'],
                 'whole_hdf5_downloaded_and_md5_verified': False, 'byte_range_header_and_length_verified': True,
                 'obs_n': len(obs), 'var_n': len(var), 'obs_columns': list(obs),
                 'possible_pairing_columns': [c for c in obs if any(k in c.lower() for k in ['sample','donor','nfkb','nfk','total_counts','tau','disease','date','batch','cluster','leiden'])],
                 'new_model_trained': False}
        (ROOT/'paired_annotation_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)

with zipfile.ZipFile(RAW/'pTau_Blots.zip') as z:
    members = [{'filename': i.filename, 'bytes': i.file_size} for i in z.infolist() if not i.is_dir()]
    (ROOT/'wb_archive_inventory.json').write_text(json.dumps(members, ensure_ascii=False, indent=2), encoding='utf-8')
    print('WB archive entries:', len(members))
