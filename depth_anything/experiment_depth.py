"""Run Depth Anything V1/V2/DA3 Small on identical, square RGB inputs."""
from __future__ import annotations
import argparse,gc,hashlib,json,platform,statistics,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download

MODELS={'v1':'LiheYoung/depth-anything-small-hf','v2':'depth-anything/Depth-Anything-V2-Small-hf','v3':'depth-anything/DA3-SMALL'}
REVISIONS={'v1':'25216a913fa218ccb7d58cce818d52b728b6c1f6','v2':'5426e4f0f36572d16453bbda7a8389317b1bef99','v3':'e08cab65ca0ec38e7826075418411ab90cab4da3'}
SOURCES={'coffee':('Rachel Michetti','CC0'),'chelsea':('Stefan van der Walt','CC0'),'camera':('Lav Varshney','CC0')}

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def inputs(root):
    from skimage import data
    out=root/'inputs';out.mkdir(parents=True,exist_ok=True);rows=[]
    for name,(author,license_) in SOURCES.items():
        a=getattr(data,name)();im=Image.fromarray(a).convert('RGB');w,h=im.size;s=min(w,h)
        box=((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2)
        im=im.crop(box).resize((518,518),Image.Resampling.BICUBIC)
        path=out/f'{name}.png';im.save(path)
        rows.append({'name':name,'original_size':[w,h],'crop':box,'size':[518,518],
          'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'author':author,'license':license_})
    write_json(out/'manifest.json',rows);return rows

def measure(fn,repeats):
    samples=[];value=None
    for _ in range(repeats):
        torch.cuda.synchronize();t=time.perf_counter();value=fn();torch.cuda.synchronize()
        samples.append((time.perf_counter()-t)*1000)
    return value,{'samples_ms':samples,'median_ms':statistics.median(samples),'min_ms':min(samples),'max_ms':max(samples)}

def near_display(raw,version):
    # V1/V2 return disparity-like relative values; DA3 returns relative depth.
    v=raw if version!='v3' else -raw
    lo,hi=np.percentile(v,[2,98])
    return np.clip((v-lo)/max(float(hi-lo),1e-12),0,1).astype(np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--root',type=Path,default=Path('depth_results'));p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--revision',help='Pinned Hugging Face commit, recorded in result JSON')
    args=p.parse_args();root=args.root;rows=inputs(root)
    if args.repeats<1:raise SystemExit('repeats must be positive')
    if not torch.cuda.is_available():raise SystemExit('CUDA GPU required for this protocol')
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    version=args.model;repo=MODELS[version]
    revision=args.revision or REVISIONS[version]
    t=time.perf_counter()
    snapshot=Path(snapshot_download(repo,revision=revision,allow_patterns=['*.json','*.safetensors']))
    download_s=time.perf_counter()-t
    t=time.perf_counter()
    if version=='v3':
        from depth_anything_3.api import DepthAnything3
        model=DepthAnything3.from_pretrained(str(snapshot),strict=True).eval().to('cuda')
        processor=None
    else:
        from transformers import AutoImageProcessor,AutoModelForDepthEstimation
        processor=AutoImageProcessor.from_pretrained(str(snapshot),use_fast=False)
        model=AutoModelForDepthEstimation.from_pretrained(str(snapshot),use_safetensors=True).eval().to('cuda')
    torch.cuda.synchronize();load_s=time.perf_counter()-t
    output={'created_at':datetime.now(timezone.utc).isoformat(),'model':repo,'revision':revision,
      'parameters':sum(v.numel() for v in model.parameters()),'weight_dtypes':sorted({str(v.dtype) for v in model.parameters()}),'download_or_cache_s':download_s,'load_to_cuda_s':load_s,
      'environment':{'python':platform.python_version(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda},
      'protocol':{'seed':42,'threads':8,'input':[1,3,518,518],'weight_dtype':'float32','autocast':'bfloat16',
       'warmups':2,'forward_repeats':args.repeats,'api_repeats':5,'tf32':False,'compile':False,
       'forward_timing':'normalized input already on CUDA; all native output heads; synchronized wall time',
       'api_timing':'PIL input already decoded in RAM; preprocessing + transfer + model + CPU output; no download/save'},'images':[]}
    rawdir=root/'raw';rawdir.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
      for row in rows:
        im=Image.open(root/'inputs'/f'{row["name"]}.png').convert('RGB')
        cpu=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255
        cpu=(cpu-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        tensor=cpu.unsqueeze(0).cuda()
        def forward():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return model(tensor.unsqueeze(1),export_feat_layers=[]) if version=='v3' else model(pixel_values=tensor)
        api_meta={}
        def api():
            if version=='v3':
                pred=model.inference([im],process_res=518)
                if not api_meta:
                    api_meta.update({k:list(getattr(pred,k).shape) if getattr(pred,k,None) is not None else None for k in ['depth','conf','extrinsics','intrinsics']})
                return pred.depth[0]
            batch=processor(images=im,do_resize=False,return_tensors='pt').to('cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):value=model(**batch).predicted_depth[0]
            return value.float().cpu().numpy()
        torch.cuda.reset_peak_memory_stats()
        first,first_time=measure(api,1)
        for _ in range(2):forward()
        raw_forward,forward_times=measure(forward,args.repeats)
        del raw_forward
        for _ in range(2):api()
        raw,api_times=measure(api,5);raw=np.asarray(raw,dtype=np.float32)
        if raw.shape!=(518,518) or not np.isfinite(raw).all():raise RuntimeError(f'invalid depth {version} {raw.shape}')
        path=rawdir/f'{version}_{row["name"]}.npy';np.save(path,raw)
        Image.fromarray((near_display(raw,version)*255).round().astype('uint8')).save(root/'raw'/f'{version}_{row["name"]}_near.png')
        result={'input':row,'first_api_for_image':first_time,'native_api_shapes':api_meta,'forward':forward_times,'api':api_times,
          'allocated_peak_mib':torch.cuda.max_memory_allocated()/1024**2,'reserved_peak_mib':torch.cuda.max_memory_reserved()/1024**2,
          'raw':{'shape':list(raw.shape),'min':float(raw.min()),'max':float(raw.max()),'median':float(np.median(raw)),
          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
        output['images'].append(result);print(json.dumps({'model':version,'image':row['name'],'forward':forward_times['median_ms'],'api':api_times['median_ms']},ensure_ascii=False),flush=True)
        write_json(root/f'{version}.json',output)
        del tensor,first,raw
    print('Saved',root/f'{version}.json',flush=True)
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
