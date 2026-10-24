import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; D=json.loads((ROOT/'ai_lab/results/pytorch_onnx.json').read_text()); OUT=ROOT/'figures/pytorch-onnx-fashionmnist.png'
fig,ax=plt.subplots(1,2,figsize=(12,5.2),constrained_layout=True)
bs=['1','32','256']; x=np.arange(3); w=.35
pt=[D['timings'][b]['pytorch_cpu']['median_ms'] for b in bs]; ort=[D['timings'][b]['onnxruntime_cpu']['median_ms'] for b in bs]
ax[0].bar(x-w/2,pt,w,label='PyTorch CPU',color='#6b8afd'); ax[0].bar(x+w/2,ort,w,label='ONNX Runtime CPU',color='#25b99a')
ax[0].set_xticks(x,[f'batch {b}' for b in bs]); ax[0].set_ylabel('Median latency (ms)'); ax[0].set_title('100 repeats after 20 warmups'); ax[0].legend(); ax[0].grid(axis='y',alpha=.25)
speed=[D['timings'][b]['ort_speedup'] for b in bs]; ax[1].bar(bs,speed,color=['#e35d6a','#25b99a','#25b99a']); ax[1].axhline(1,color='#333',lw=1)
ax[1].set_xlabel('Batch size'); ax[1].set_ylabel('PyTorch latency / ORT latency'); ax[1].set_title('ONNX Runtime speedup'); ax[1].grid(axis='y',alpha=.25)
for i,v in enumerate(speed): ax[1].text(i,v+.07,f'{v:.2f}×',ha='center',weight='bold')
fig.suptitle('PyTorch → ONNX: identical 10,000 predictions, different CPU speed',fontsize=16,weight='bold')
fig.text(.5,.01,f"max |logit diff| = {D['parity']['max_abs_logit_diff']:.2e} · accuracy = {D['parity']['pytorch_accuracy']*100:.2f}% · changed predictions = 0",ha='center',color='#555')
OUT.parent.mkdir(parents=True,exist_ok=True); fig.savefig(OUT,dpi=180,facecolor='white'); print(OUT)
