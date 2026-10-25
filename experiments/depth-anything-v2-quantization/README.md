# Depth Anything V2 SmallをFP32・INT8・INT4で比べる

同じV2 Smallの重みからONNXモデルを作り、CPUとCUDAで推論します。DA-2Kの近遠判定、NYUv2とMPI Sintelの深度評価、ファイル容量、推論時間を別々に記録します。

`evidence/` は記事で使った測定値です。`results*/` は手元で再実行したときの出力になります。元のデータセット、モデル重み、予測配列は実行時に取得・生成します。

## 用意するもの

必須なのはWindows 11、Python 3.12、Git、インターネット接続です。試験機はCore i7-14700F、RAM 32 GB。GPUを使わない部分から実行できます。

CUDAの比較には対応するNVIDIA GPUとドライバーが必要です。試験機はRTX 5070 Ti 16 GB、NVIDIAドライバー610.88で、ONNX RuntimeのCUDA 13 / cuDNN 9構成を使いました。CPU版とGPU版のONNX Runtimeは別の仮想環境へ入れます。

環境と全データ・生成物を含め、空き容量は20 GB程度を見込んでください。DA-2KのZIPは約681 MB、NYUv2の原本は約2.97 GB。Sintelは必要なZIP内メンバーだけをHTTP Rangeで取得し、今回の64画像では約135 MBを転送しました。依存パッケージ、とくにGPU用ライブラリの取得にも数GBかかります。通常の実行に管理者権限や再起動は要りません。Pythonやドライバーを新規導入する場合は別です。

## モデルを作る

以下はPowerShell用です。最初の`python`がPython 3.12を指すことを確認してください。作業場所は取得したリポジトリ内のこのフォルダーです。

```powershell
git clone --branch depth-anything-v2-quantization --single-branch https://github.com/matrizea/yuyuyuroom-ai-experiments.git ai-experiments
cd ai-experiments/experiments/depth-anything-v2-quantization
python --version
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/DepthAnything/Depth-Anything-V2.git vendor/Depth-Anything-V2
git -C vendor/Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
.venv/Scripts/python.exe scripts/build_models.py
```

公式実装のcommitを固定するのは、モデル構造や前後処理の変更を混ぜないためです。`build_models.py`は作者のSmall checkpointをダウンロードし、変更していない公式forwardをONNXへ変換します。乱数seed42の入力でPyTorchとONNX出力を照合し、ずれが大きい場合は停止します。

正常なら`models/metadata.json`と4つのONNXファイルが作られ、最後に`export_validation`と`quantized_matmuls: 48`が表示されます。記事の元checkpoint SHA-256は`715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`です。

入力は518×518へのCUBIC直接リサイズで、cropはしません。縦横比を保つ公式の標準前処理とは異なります。画像を正方形へ揃えたこの比較と、公式ベンチマークの数値を混ぜないでください。

## まず手元の1枚を比べる

```powershell
.venv/Scripts/python.exe scripts/compare_image.py --image "C:/photos/example.jpg" --output-dir results/custom
```

`comparison.jpg`と`comparison.json`、加工前の深度配列が保存されます。正解深度がない画像なので、ここでは正答率を測りません。表示される誤差はFP32出力との差です。

## 近い点・遠い点を判定する

```powershell
.venv/Scripts/python.exe scripts/prepare_data.py
.venv/Scripts/python.exe scripts/evaluate.py --limit 2 --output-dir results_smoke --skip-benchmark
.venv/Scripts/python.exe scripts/evaluate.py --threads 8 --skip-benchmark
.venv/Scripts/python.exe scripts/evaluate.py --threads 8 --benchmark-only
.venv/Scripts/python.exe scripts/make_report.py
```

最初は2画像の試行で、モデルの読み込みと前処理を確認します。通常評価はDA-2Kの8カテゴリから各32画像、計256画像・507ペアです。同じ画像内のペアはまとめて再抽出し、FP32からの正答率差の95%信頼区間を計算します。

速度の測定時は、ほかのCPU実験を止めてください。表の時間はモデルを読み込んだ後の`session.run`で、読込・前後処理・保存を含みません。通常の256画像評価は試験機で数分程度ですが、取得時間とPCの負荷で変わります。

## GPUではFP16も入れて比べる

```powershell
python -m venv .venv_gpu
.venv_gpu/Scripts/python.exe -m pip install -r requirements-gpu-lock.txt
.venv_gpu/Scripts/python.exe -c "import sys; sys.path.insert(0,'scripts'); import runtime_init; import onnxruntime as ort; ort.preload_dlls(directory=''); print(ort.get_available_providers())"
.venv_gpu/Scripts/python.exe scripts/build_gpu_models.py
.venv_gpu/Scripts/python.exe scripts/benchmark_gpu.py
```

Provider一覧に`CUDAExecutionProvider`が必要です。スクリプトはCUDAが初期化できないと停止します。ただし、CUDAが利用できても全演算がGPU上とは限りません。`results_gpu/benchmark.json`の`profile.node_executions_by_provider`も確認します。

今回のdynamic INT8には、CPU側の`DynamicQuantizeLinear`と`MatMulInteger`が各48個ありました。GPUとCPUの間で中間値を転送するため、CPUで速かった設定がCUDAでは遅くなっています。`MatMulNBits`を使うW8A16/W4A16は別のモデルです。演算名だけからネイティブ整数Tensor Core使用を断定できません。

GPUはウォームアップ20回、計測100回、TF32無効。CPU上の入力・出力を使う時間と、入出力をGPU上へ置くI/O Bindingの時間を両方保存します。後者も、内部にCPU演算が残っている場合の転送まで消す仕組みではありません。

## 画素ごとの正解深度で評価する

```powershell
.venv/Scripts/python.exe scripts/prepare_dense_data.py
.venv/Scripts/python.exe scripts/prepare_sintel_data.py
.venv/Scripts/python.exe scripts/evaluate_dense.py --manifest data_dense/manifest.json --provider CPU --output-dir results_dense_nyu_cpu
.venv/Scripts/python.exe scripts/evaluate_dense.py --manifest data_dense/sintel_manifest.json --provider CPU --output-dir results_dense_sintel_full_cpu
```

NYUv2は公式test 654画像からseed42で128画像を選びます。RGBへ投影したKinect実測値`rawDepths`と、欠損を補完した`depths`を別に集計します。補完された全画素を、独立した実測正解とは扱いません。有効範囲は有限な`0 < depth < 10 m`、cropなしです。

Sintelは合成映像23シーケンスから計64画像を選び、レンダリングされた深度を使います。有効範囲は有限な`0 < depth < 10000 m`。この上限は実験側の設定で、巨大な背景相当値を除き、それ以外の深度の最大値約3364 mを残すためです。公式標準の必須条件ではありません。配布深度がカメラ軸方向か光線方向かを一次資料で確定できなかったため、値は変換せず使用しています。

GPUのdense評価は、同じmanifestに対して次を実行します。NYUv2の場合はmanifestと出力先を差し替えます。

```powershell
.venv_gpu/Scripts/python.exe scripts/evaluate_dense.py --manifest data_dense/sintel_manifest.json --provider CUDA --output-dir results_dense_sintel_full_gpu --models fp32 int8_dynamic int4_rtn int4_fast fp16=models_gpu/fp16.onnx int8_w8a16=models_gpu/int8_w8a16.onnx int4_w4a16=models_gpu/int4_w4a16.onnx
```

このモデルは相対的な逆深度を返すので、そのままメートルの誤差を計算できません。各画像のGTを使って逆深度の倍率と切片を合わせた後、AbsRel、RMSE、δ1を計算します。GTを使った整合後の形状評価であり、未知の写真から絶対距離を当てる精度ではありません。

maskは全モデル共通のGTだけから決め、予測が苦手な画素を後から落としません。整合した逆深度が非正値などになったときは`1 / max_depth`で下限処理して距離へ戻し、その画素も評価に残してclip率を記録します。主要値は画像ごとの指標の平均です。信頼区間は同じscene/sequenceをまとめて再抽出するものを優先します。

## 保存した結果から図を作る

```powershell
.venv/Scripts/python.exe scripts/plot_evidence.py
.venv/Scripts/python.exe scripts/validate_evidence.py
```

`article_figures/`へ、公開した数値記録を使ったCPU/GPUグラフが出ます。元のRGB画像やGTファイルは含みません。検算スクリプトはCSVから正答数・誤差平均を再集計し、各回の推論時間から中央値を計算し直します。

## 途中で止まった場合

- `No module named depth_anything_v2`：上の公式repoを`vendor/Depth-Anything-V2`へ取得したか確認します。モデル重みだけでは実装は揃いません。
- DLLの読み込みに失敗する：指定したvenvのPythonとrequirementsを使っているか確認します。`runtime_init.py`はWindows用の同梱MSVC DLLを先に読み込みます。
- CUDAがない、またはCUDA初期化エラー：CPU用とGPU用venvを取り違えていないか、Provider一覧とドライバーを確認します。実行ログだけでGPU計算と判断しません。
- `Checkpoint configuration differs`：同じ出力フォルダーに別条件を混ぜています。既存結果を残し、別の`--output-dir`を指定します。
- データ取得が止まった：既存ファイルのハッシュ・途中ファイルを確認します。データを別のsplitへ勝手に替えず、元のエラーを確認して再実行してください。

## 公式資料

- [Depth Anything V2公式実装](https://github.com/DepthAnything/Depth-Anything-V2)、[Small checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)、[論文](https://arxiv.org/abs/2406.09414)
- [ONNX Runtimeの量子化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)、[CUDA Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [DA-2K](https://huggingface.co/datasets/depth-anything/DA-2K)、[NYUv2作者配布元](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)、[MPI Sintel depth](https://sintel.is.tue.mpg.de/depth)

Small checkpointと公式実装はApache-2.0です。NYUv2とMPI Sintelについて、確認した作者配布資料から明示的な再配布ライセンスは確認できていません。映画Sintel自体のCC BY 3.0とMPIの変更された評価データを同じものとして扱わず、元データはそれぞれの公式配布元から取得してください。
