import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=json.loads((ROOT/'ai_lab/results/weight_quant_sim.json').read_text());OUT=ROOT/'figures/int8-int4-weight-quant.png'
labels=['FP32','INT8\nper-tensor','INT8\nper-channel','INT4\nper-tensor','INT4\nper-channel']; rows=D['results']; acc=[D['fp32']['accuracy']*100]+[x['accuracy']*100 for x in rows]; changed=[0]+[x['changed_predictions'] for x in rows]; sizes=[D['fp32']['parameter_bytes']/1024]+[x['packed_parameter_bytes_theoretical']/1024 for x in rows]
fig,ax=plt.subplots(1,3,figsize=(14,5.2),constrained_layout=True); colors=['#607086','#6b8afd','#25b99a','#e35d6a','#f3b641']
ax[0].bar(labels,acc,color=colors);ax[0].set_ylim(78,88);ax[0].set_ylabel('Test accuracy (%)');ax[0].set_title('10,000 FashionMNIST images');ax[0].tick_params(axis='x',rotation=25);ax[0].grid(axis='y',alpha=.25)
for i,v in enumerate(acc):ax[0].text(i,v+.15,f'{v:.2f}%',ha='center',fontsize=9,weight='bold')
ax[1].bar(labels,changed,color=colors);ax[1].set_ylabel('Changed top-1 predictions');ax[1].set_title('Versus FP32');ax[1].tick_params(axis='x',rotation=25);ax[1].grid(axis='y',alpha=.25)
ax[2].bar(labels,sizes,color=colors);ax[2].set_ylabel('Theoretical parameter KiB');ax[2].set_title('Packed weights + FP32 bias');ax[2].tick_params(axis='x',rotation=25);ax[2].grid(axis='y',alpha=.25)
fig.suptitle('Weight-only fake quantization: granularity matters at INT4',fontsize=16,weight='bold');OUT.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUT,dpi=180,facecolor='white');print(OUT)
