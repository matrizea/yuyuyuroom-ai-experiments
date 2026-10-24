#!/usr/bin/env python3
"""Train FashionMNIST CNN, export to ONNX, and verify ONNX Runtime parity/speed."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiment_fashion_cnn import SmallCNN


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ai_lab" / "data"
OUT = ROOT / "ai_lab" / "results" / "pytorch_onnx.json"
MODEL_DIR = ROOT / "ai_lab" / "models"
PT_PATH = MODEL_DIR / "fashion_cnn_state_dict.pt"
ONNX_PATH = MODEL_DIR / "fashion_cnn_dynamic.onnx"


def bench(fn, warmup=20, repeats=100):
    for _ in range(warmup): fn()
    values=[]
    for _ in range(repeats):
        t0=time.perf_counter(); fn(); values.append((time.perf_counter()-t0)*1000)
    return {"median_ms":statistics.median(values),"mean_ms":statistics.mean(values),
            "p95_ms":float(np.percentile(values,95)),"repeats":repeats}


def main():
    torch.manual_seed(42); np.random.seed(42)
    tfm=transforms.ToTensor()
    train=datasets.FashionMNIST(DATA,train=True,download=True,transform=tfm)
    test=datasets.FashionMNIST(DATA,train=False,download=True,transform=tfm)
    train_loader=DataLoader(train,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=0)
    model=SmallCNN().cuda(); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    epochs=[]
    for epoch in range(1,4):
        model.train(); t0=time.perf_counter(); seen=correct=0; loss_sum=0.0
        for x,y in train_loader:
            x=x.cuda(); y=y.cuda(); opt.zero_grad(set_to_none=True); z=model(x); loss=loss_fn(z,y); loss.backward(); opt.step()
            seen+=len(y); correct+=int((z.argmax(1)==y).sum()); loss_sum+=float(loss)*len(y)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-t0
        epochs.append({"epoch":epoch,"seconds":elapsed,"loss":loss_sum/seen,"train_accuracy":correct/seen})
        print(f"epoch {epoch}: {elapsed:.3f}s loss={loss_sum/seen:.4f}",flush=True)

    model=model.cpu().eval(); MODEL_DIR.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),PT_PATH)
    example=torch.zeros(1,1,28,28)
    torch.onnx.export(model,example,ONNX_PATH,input_names=["image"],output_names=["logits"],
                      dynamic_axes={"image":{0:"batch"},"logits":{0:"batch"}},opset_version=18,dynamo=False)
    checked=onnx.load(ONNX_PATH); onnx.checker.check_model(checked)
    sess=ort.InferenceSession(str(ONNX_PATH),providers=["CPUExecutionProvider"])

    all_x=torch.stack([test[i][0] for i in range(len(test))]); all_y=np.array([test[i][1] for i in range(len(test))])
    pt_logits=[]; ort_logits=[]
    with torch.inference_mode():
        for start in range(0,len(test),512):
            batch=all_x[start:start+512]; pt_logits.append(model(batch).numpy())
            ort_logits.append(sess.run(["logits"],{"image":batch.numpy()})[0])
    pt_logits=np.vstack(pt_logits); ort_logits=np.vstack(ort_logits)
    diff=np.abs(pt_logits-ort_logits); pt_pred=pt_logits.argmax(1); ort_pred=ort_logits.argmax(1)
    parity={"max_abs_logit_diff":float(diff.max()),"mean_abs_logit_diff":float(diff.mean()),
            "top1_agreement":float(np.mean(pt_pred==ort_pred)),"pytorch_accuracy":float(np.mean(pt_pred==all_y)),
            "onnxruntime_accuracy":float(np.mean(ort_pred==all_y)),"changed_predictions":int(np.sum(pt_pred!=ort_pred))}

    timings={}
    for bs in (1,32,256):
        a=np.asarray(all_x[:bs].numpy(),dtype=np.float32); t=torch.from_numpy(a)
        with torch.inference_mode():
            pt=bench(lambda:model(t)); ortr=bench(lambda:sess.run(["logits"],{"image":a}))
        for result in (pt,ortr): result["images_per_s"]=bs/(result["median_ms"]/1000)
        timings[str(bs)]={"pytorch_cpu":pt,"onnxruntime_cpu":ortr,"ort_speedup":pt["median_ms"]/ortr["median_ms"]}
        print(bs,timings[str(bs)],flush=True)

    output={"environment":{"torch":torch.__version__,"onnx":onnx.__version__,"onnxruntime":ort.__version__,
                           "providers":ort.get_available_providers(),"gpu":torch.cuda.get_device_name(0)},
            "protocol":{"seed":42,"epochs":3,"train_batch":256,"opset":18,"dynamic_batch":True,
                        "test_samples":len(test),"benchmark_warmup":20,"benchmark_repeats":100},
            "training":epochs,"parity":parity,"timings":timings,
            "artifacts":{"state_dict":str(PT_PATH),"state_dict_bytes":PT_PATH.stat().st_size,
                         "onnx":str(ONNX_PATH),"onnx_bytes":ONNX_PATH.stat().st_size,
                         "onnx_ir_version":checked.ir_version,"onnx_nodes":len(checked.graph.node)}}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"parity":parity,"artifacts":output["artifacts"]},ensure_ascii=False,indent=2)); print(OUT)

if __name__=="__main__": main()
