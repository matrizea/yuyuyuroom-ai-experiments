"""Count checkpoint scalar values independently of model-card estimates."""
import hashlib,json,math
from pathlib import Path
from safetensors import safe_open
from huggingface_hub import snapshot_download
from experiment_depth import MODELS,REVISIONS
rows=[]
for version,repo in MODELS.items():
    path=Path(snapshot_download(repo,revision=REVISIONS[version],local_files_only=True))/'model.safetensors'
    with safe_open(str(path),framework='pt') as weights:
        count=sum(math.prod(weights.get_slice(key).get_shape()) for key in weights.keys())
    rows.append({'version':version,'model':repo,'revision':REVISIONS[version],
      'checkpoint_scalars':count,'file_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if version=='v3':
        from depth_anything_3.api import DepthAnything3
        model=DepthAnything3.from_pretrained(str(path.parent),strict=True)
        rows[-1]['strict_load_passed']=True
        assert sum(v.numel() for v in model.parameters())==count
        del model
out=Path(__file__).resolve().parent/'results'/'checkpoint_audit.json'
out.write_text(json.dumps(rows,indent=2),encoding='utf-8');print(out.read_text())
