# Depth Anything V1・V2・V3の比較

同じ3枚の写真をSmall版へ入力し、深度画像、処理時間、PyTorchのメモリ使用量を保存します。V1・V2はTransformers版、V3は公式のDA3 APIを使います。

## 環境を作る

WSL 2、CUDAが使えるNVIDIA GPU、Git、uvが必要です。Pythonは3.12を使います。試したのはRTX 5070 Ti、PyTorch 2.13.0+cu130です。空きは15 GB程度を見込みます。

uvがない場合は、[公式の導入方法](https://docs.astral.sh/uv/getting-started/installation/)か、次の専用環境で用意できます。

```bash
python3 -m venv ~/.venvs/depth-tools
~/.venvs/depth-tools/bin/python -m pip install uv
export PATH="$HOME/.venvs/depth-tools/bin:$PATH"
```

以下はWSLのターミナルで実行します。仮想環境とモデルのキャッシュはLinux側へ置く構成です。

```bash
git clone https://github.com/matrizea/yuyuyuroom-ai-experiments.git ~/ai-experiments
cd ~/ai-experiments/depth_anything
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git vendor/Depth-Anything-3
git -C vendor/Depth-Anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
uv pip install -e vendor/Depth-Anything-3 --no-deps
python -c 'import torch; print(torch.cuda.is_available())'
```

最後が `True` なら比較を実行します。必要なモデル重み3個は初回に取得します。合計は約320 MiBです。Pythonパッケージの取得は別に数GBあります。

## 実行する

```bash
python run_all.py
python validate_results.py --root results
python make_report.py
python -m unittest discover -s . -p test_depth.py -v
```

`run_all.py v3` のように指定すれば、1つだけ実行できます。既存の同名結果は更新するので、比較条件を変える前に `results/` を別名で保管してください。

| 保存先 | 内容 |
|---|---|
| `results/inputs/` | 中央を切り出した518×518画像とハッシュ |
| `results/raw/` | 加工前の深度NPY、近いほど白い表示用PNG |
| `results/v1.json` など | 各回の時間、条件、入力と出力のハッシュ |
| `figures/` | 入力と3モデルの比較図 |
| `viewer.html` | 保存済みの予測を切り替えて見る画面 |
| `evidence/` | 記事の表に使った測定JSON。再実行しても更新されない |

ブラウザーで見るには、同じフォルダーでローカルサーバーを起動します。

```bash
python -m http.server 8765 --bind 127.0.0.1
```

同じPCで `http://127.0.0.1:8765/viewer.html` を開きます。終了はターミナルでCtrl+C。

## 数字と画像の読み方

入力は同じ写真を中央で正方形に切り出して518×518へリサイズしています。重みはFP32、CUDAのautocastはBF16、CPUは8スレッドです。モデルのrevisionは `experiment_depth.py` に固定しています。

forwardの時間は前処理済み入力がGPU上にある状態、APIの時間はデコード済みPIL画像からCPU上の深度配列を得るまでです。画像読込・保存、モデルの取得・ロードは時間に含めません。ウォームアップ2回の後、写真ごとにforwardを10回、APIを5回測ります。DA3は深度だけでなくカメラ関連の出力も計算します。

V1・V2は大きいほど手前、DA3 Smallは大きいほど奥です。表示用PNGだけDA3の符号を反転し、各画像の2〜98パーセンタイルで正規化します。出力にメートルの単位はありません。正解深度のない3枚なので、見た目から深度精度の順位を決める実験ではありません。

`audit_weights.py` はキャッシュにある重みのSHA-256と要素数を検査します。モデルカードの概数と別に、実際に使ったチェックポイントを数えるためのものです。

## DA3の読み込みで困った場合

この依存一覧は単眼推論経路を試した構成です。公式Web UIや3DGSの全機能は含みません。`gsplat` の描画依存に関する警告は出ますが、この深度推論では使っていません。SDK本体の導入には `--no-deps` を使い、必要な依存は上の一覧で揃えます。

`addict` の不足は読み込みエラーになるので依存一覧へ含めています。直接 `forward()` を呼ぶところでは `export_feat_layers=[]` を明示しています。SDKの `inference()` を使う経路とは引数の扱いが違いました。

## 写真とモデル

写真はscikit-imageのCC0データです。Coffee: Rachel Michetti、Chelsea: Stefan van der Walt、Camera: Lav Varshney。取得・切り出し方法はコード内にあります。[元データの説明](https://scikit-image.org/docs/stable/api/skimage.data.html)も参照してください。

今回取得するV1 Small、V2 Small、DA3 SmallのモデルカードはいずれもApache-2.0です。モデル重みとSDK本体はこのリポジトリに同梱していません。

- [V1](https://huggingface.co/LiheYoung/depth-anything-small-hf)
- [V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- [V3](https://huggingface.co/depth-anything/DA3-SMALL)
