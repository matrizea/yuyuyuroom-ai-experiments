#!/usr/bin/env python3
"""Weight-only symmetric INT8/INT4 fake-quantization of the trained FashionMNIST CNN."""

from __future__ import annotations
import copy,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from experiment_fashion_cnn import SmallCNN

ROOT=Path(__file__).resolve().parents[1]; STATE=ROOT/'ai_lab/models/fashion_cnn_state_dict.pt'; OUT=ROOT/'ai_lab/results/weight_quant_sim.json'; DATA=ROOT/'ai_lab/data'

def qdq(w,bits,per_channel):
    qmax=2**(bits-1)-1
    if per_channel and w.ndim>=2:
        dims=tuple(range(1,w.ndim)); scale=w.abs().amax(dim=dims,keepdim=True)/qmax
    else: scale=w.abs().max()/qmax
    scale=torch.clamp(scale,min=torch.finfo(torch.float32).eps)
    q=torch.clamp(torch.round(w/scale),-qmax,qmax); return q*scale,float(torch.mean((w-q*scale)**2)),int(q.numel())

def logits(model,loader):
    out=[]; labels=[]; model.eval()
    with torch.inference_mode():
        for x,y in loader: out.append(model(x).numpy()); labels.append(y.numpy())
    return np.vstack(out),np.concatenate(labels)

def main():
    base=SmallCNN(); base.load_state_dict(torch.load(STATE,map_location='cpu',weights_only=True)); test=datasets.FashionMNIST(DATA,train=False,download=True,transform=transforms.ToTensor()); loader=DataLoader(test,batch_size=512,shuffle=False)
    ref,y=logits(base,loader); refp=ref.argmax(1); params=dict(base.named_parameters()); total=sum(p.numel() for p in params.values()); weight_n=sum(p.numel() for n,p in params.items() if n.endswith('weight')); bias_n=total-weight_n
    results=[]
    for bits in (8,4):
      for pc in (False,True):
        m=copy.deepcopy(base); mses=[]
        with torch.no_grad():
          for name,p in m.named_parameters():
            if name.endswith('weight'):
              v,mse,_=qdq(p,bits,pc); p.copy_(v); mses.append({"name":name,"mse":mse})
        z,_=logits(m,loader); pred=z.argmax(1); diff=np.abs(ref-z)
        packed_bytes=(weight_n*bits+7)//8+bias_n*4
        row={"bits":bits,"granularity":"per_output_channel" if pc else "per_tensor","accuracy":float(np.mean(pred==y)),"changed_predictions":int(np.sum(pred!=refp)),"top1_agreement":float(np.mean(pred==refp)),"max_abs_logit_diff":float(diff.max()),"mean_abs_logit_diff":float(diff.mean()),"packed_parameter_bytes_theoretical":packed_bytes,"weight_mse":mses}
        results.append(row); print(row,flush=True)
    output={"protocol":{"type":"weight-only symmetric fake quantization","activation":"FP32","bias":"FP32","weights":weight_n,"biases":bias_n,"test_samples":len(test),"note":"weights are dequantized back to FP32 for inference; no integer kernel speed claim"},"fp32":{"accuracy":float(np.mean(refp==y)),"parameter_bytes":total*4,"state_dict_bytes":STATE.stat().st_size},"results":results}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8'); print(OUT)
if __name__=='__main__': main()
