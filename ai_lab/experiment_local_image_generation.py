#!/usr/bin/env python3
"""Generate SD-Turbo images locally and benchmark steps/reproducibility."""
from __future__ import annotations
import hashlib,json,time,statistics
from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]; ODIR=ROOT/'ai_lab/generated_images'; OUT=ROOT/'ai_lab/results/local_image_generation.json'; GRID=ROOT/'figures/sd-turbo-local-steps.png'
MODEL='stabilityai/sd-turbo'; PROMPT='A small university AI laboratory desk, a black GPU workstation beside notebooks with neural network diagrams, warm morning light, realistic editorial photograph, no text, no logo'

def digest(im): return hashlib.sha256(im.tobytes()).hexdigest()
def main():
 ODIR.mkdir(parents=True,exist_ok=True); torch.manual_seed(42); t0=time.perf_counter()
 pipe=AutoPipelineForText2Image.from_pretrained(MODEL,torch_dtype=torch.float16,variant='fp16'); pipe=pipe.to('cuda'); load_s=time.perf_counter()-t0
 # One unmeasured warmup pays CUDA/kernel initialization before comparing step counts.
 pipe(PROMPT,num_inference_steps=1,guidance_scale=0.0,height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(999))
 rows=[]; images=[]
 for steps in (1,2,4,8):
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();samples=[];im=None
  for _ in range(5):
   g=torch.Generator(device='cuda').manual_seed(42);torch.cuda.synchronize();t=time.perf_counter();im=pipe(PROMPT,num_inference_steps=steps,guidance_scale=0.0,height=512,width=512,generator=g).images[0];torch.cuda.synchronize();samples.append(time.perf_counter()-t)
  elapsed=statistics.median(samples);p=ODIR/f'sd_turbo_steps{steps}_seed42.png';im.save(p);images.append((steps,im.copy()));rows.append({'steps':steps,'seed':42,'median_seconds':elapsed,'seconds_samples':samples,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'sha256_pixels':digest(im),'file':str(p),'file_bytes':p.stat().st_size});print(rows[-1],flush=True)
 # Repeat 4-step generation with the same and a different seed.
 for seed,label in ((42,'repeat'),(43,'different_seed')):
  g=torch.Generator(device='cuda').manual_seed(seed);im=pipe(PROMPT,num_inference_steps=4,guidance_scale=0.0,height=512,width=512,generator=g).images[0];p=ODIR/f'sd_turbo_steps4_seed{seed}_{label}.png';im.save(p)
  rows.append({'steps':4,'seed':seed,'label':label,'sha256_pixels':digest(im),'file':str(p),'file_bytes':p.stat().st_size})
 # Presentation grid uses only generated pixels plus labels.
 canvas=Image.new('RGB',(1024,560),'white');d=ImageDraw.Draw(canvas)
 for i,(steps,im) in enumerate(images):
  x=(i%2)*512;y=(i//2)*280;canvas.paste(im.resize((256,256)),(x,y+24));d.text((x+270,y+110),f'{steps} step'+('s' if steps>1 else ''),fill='black');d.text((x+270,y+135),f"{next(r['median_seconds'] for r in rows if r.get('steps')==steps and 'median_seconds' in r):.3f} s median",fill='#444')
 GRID.parent.mkdir(parents=True,exist_ok=True);canvas.save(GRID)
 output={'model':MODEL,'prompt':PROMPT,'load_seconds':load_s,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'dtype':'float16','resolution':[512,512],'guidance_scale':0.0,'runs':rows,'same_seed_pixel_identical':rows[2]['sha256_pixels']==rows[4]['sha256_pixels'],'different_seed_pixel_different':rows[2]['sha256_pixels']!=rows[5]['sha256_pixels'],'grid':str(GRID)}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(output,ensure_ascii=False,indent=2));
if __name__=='__main__':main()
