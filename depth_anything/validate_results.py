"""Validate raw arrays, hashes, repetitions and the table against saved evidence."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results'));a=p.parse_args()
for v in ['v1','v2','v3']:
    d=json.loads((a.root/f'{v}.json').read_text())
    assert len(d['images'])==3 and d['weight_dtypes']==['torch.float32']
    assert d['protocol']['input']==[1,3,518,518]
    for row in d['images']:
        name=row['input']['name'];path=a.root/'raw'/f'{v}_{name}.npy'
        raw=np.load(path,allow_pickle=False)
        assert raw.shape==(518,518) and np.isfinite(raw).all()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['raw']['sha256']
        assert hashlib.sha256((a.root/'inputs'/f'{name}.png').read_bytes()).hexdigest()==row['input']['sha256']
        for key,count in [('forward',10),('api',5)]:
            assert len(row[key]['samples_ms'])==count
            assert statistics.median(row[key]['samples_ms'])==row[key]['median_ms']
        assert row['allocated_peak_mib']>0
    print(v,'PASS: 3 images, hashes, shapes, 30 forward + 15 API timings')
