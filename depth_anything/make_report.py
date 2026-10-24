"""Build figures and a local comparison viewer from actual saved predictions."""
import hashlib,html,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root=Path(__file__).resolve().parent;out=root/'results';figdir=root/'figures';figdir.mkdir(exist_ok=True)
data={v:json.loads((out/f'{v}.json').read_text()) for v in ['v1','v2','v3']}
summary=[]
for version,d in data.items():
    summary.append({'version':version,'model':d['model'],'revision':d['revision'],'parameters':d['parameters'],
      'weight_dtypes':d['weight_dtypes'],'load_s':d['load_to_cuda_s'],
      'forward_median_ms':statistics.median(t for row in d['images'] for t in row['forward']['samples_ms']),
      'api_median_ms':statistics.median(t for row in d['images'] for t in row['api']['samples_ms']),
      'peak_allocated_mib':max(row['allocated_peak_mib'] for row in d['images']),
      'peak_reserved_mib':max(row['reserved_peak_mib'] for row in d['images']),
      'raw_ranges':{row['input']['name']:[row['raw']['min'],row['raw']['max']] for row in d['images']},
      'native_api_shapes':d['images'][0]['native_api_shapes']})
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

fig,axes=plt.subplots(3,4,figsize=(14,11.2),dpi=150)
for i,name in enumerate(['coffee','chelsea','camera']):
    axes[i,0].imshow(Image.open(out/'inputs'/f'{name}.png'))
    for j,v in enumerate(['v1','v2','v3'],1):axes[i,j].imshow(Image.open(out/'raw'/f'{v}_{name}_near.png'),cmap='gray',vmin=0,vmax=255)
    for j in range(4):
        axes[i,j].axis('off')
        if i==0:axes[i,j].set_title(['Input (518 x 518)','V1 Small','V2 Small','DA3 Small'][j],fontsize=15)
    axes[i,0].text(.02,.04,name,transform=axes[i,0].transAxes,color='white',bbox={'facecolor':'black','alpha':.65,'edgecolor':'none'})
fig.suptitle('Depth Anything: identical images, three generations',fontsize=19)
fig.text(.5,.025,'White = nearer. Per-image 2–98 percentile normalization; DA3 sign reversed. Not metric distance or an accuracy benchmark.',ha='center',fontsize=10)
fig.tight_layout(rect=(.01,.05,.99,.95));fig.savefig(figdir/'depth-anything-v1-v2-v3-comparison.png');plt.close(fig)

fig,axes=plt.subplots(1,4,figsize=(14,4),dpi=150)
for j,v in enumerate(['input','v1','v2','v3']):
    path=out/'inputs'/'coffee.png' if v=='input' else out/'raw'/f'{v}_coffee_near.png'
    axes[j].imshow(Image.open(path),cmap='gray',vmin=0,vmax=255);axes[j].set_title(v.upper());axes[j].axis('off')
fig.tight_layout();fig.savefig(figdir/'depth-anything-coffee-comparison.png');plt.close(fig)

rows=''.join(f'<tr><td>{r["version"].upper()}</td><td>{r["parameters"]/1e6:.2f} M</td><td>{r["forward_median_ms"]:.2f}</td><td>{r["api_median_ms"]:.2f}</td><td>{r["peak_allocated_mib"]:.1f}</td></tr>' for r in summary)
viewer='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Depth Anything V1 / V2 / V3 比較</title><style>
body{font:17px/1.7 system-ui,sans-serif;background:#f2f4f6;color:#17212c;margin:0;padding:24px}main{max-width:1600px;margin:auto}h1{font-size:30px;margin:0 0 8px}button{font:inherit;padding:8px 24px;margin:6px 5px 18px 0;border:1px solid #87929e;background:white;border-radius:5px;cursor:pointer}button[aria-pressed=true]{background:#20394d;color:white}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0;padding:8px;background:white;border-radius:8px}img{display:block;width:100%;height:auto}figcaption{font-weight:600}table{border-collapse:collapse;background:white;margin-top:20px}td,th{padding:10px 22px;border:1px solid #ccd4dc;text-align:right}th:first-child,td:first-child{text-align:left}.note{max-width:1050px}footer{font-size:13px;color:#475569;margin-top:15px}@media(max-width:700px){body{padding:12px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}table{font-size:12px}td,th{padding:7px}}
</style><main><h1>Depth Anything V1 / V2 / V3を同じ写真で動かす</h1>
<p class="note">Small版・518×518・RTX 5070 Ti。白いほど手前です。画像ごとに明るさを正規化しているため、白の強さを距離の差としては比べられません。</p>
<nav aria-label="比較する写真"><button data-photo="coffee" aria-pressed="true">コーヒー</button><button data-photo="chelsea" aria-pressed="false">猫</button><button data-photo="camera" aria-pressed="false">カメラ</button></nav>
<div class="grid"><figure><figcaption>入力画像</figcaption><img id="input" src="results/inputs/coffee.png" alt="中央を切り出した入力写真"></figure>
<figure><figcaption>V1 Small</figcaption><img id="v1" src="results/raw/v1_coffee_near.png" alt="V1の相対的な奥行き推定"></figure>
<figure><figcaption>V2 Small</figcaption><img id="v2" src="results/raw/v2_coffee_near.png" alt="V2の相対的な奥行き推定"></figure>
<figure><figcaption>DA3 Small</figcaption><img id="v3" src="results/raw/v3_coffee_near.png" alt="V3の相対的な奥行き推定"></figure></div>
<table><thead><tr><th>モデル</th><th>パラメーター</th><th>forward [ms]</th><th>前後処理込み [ms]</th><th>peak allocated [MiB]</th></tr></thead><tbody>ROWS</tbody></table>
<p class="note">forwardは入力がGPU上にある状態の計算時間。前後処理込みはメモリー上の画像からCPU上の出力配列までです。重みの読み込み・ダウンロード・ファイル保存は含みません。V3は深度以外の出力も計算します。</p>
<footer>入力: scikit-imageのCC0写真。Coffee: Rachel Michetti / Chelsea: Stefan van der Walt / Camera: Lav Varshney。予測配列は実モデルの出力を保存したものです。</footer>
</main><script>document.querySelectorAll('button[data-photo]').forEach(b=>b.addEventListener('click',()=>{const n=b.dataset.photo;document.querySelector('#input').src='results/inputs/'+n+'.png';['v1','v2','v3'].forEach(v=>document.getElementById(v).src='results/raw/'+v+'_'+n+'_near.png');document.querySelectorAll('button[data-photo]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));}));</script></html>'''.replace('ROWS',rows)
(root/'viewer.html').write_text(viewer,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
