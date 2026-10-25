# Depth Anything V2 SmallをFP32・FP16・INT8・INT4・FP8で比べる

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

## FP8を同じ条件のFP16と比べる

上のCPU・GPU環境とモデルを用意した後、次の順でデータを準備します。追加の依存パッケージは不要です。校正用の準備はh5pyを含むCPU環境、モデル生成と推論はGPU環境で実行します。校正スクリプトは重複確認のため、3つの評価manifestとRGB画像も読みます。

```powershell
.venv/Scripts/python.exe scripts/prepare_data.py
.venv/Scripts/python.exe scripts/prepare_dense_data.py
.venv/Scripts/python.exe scripts/prepare_sintel_data.py
.venv/Scripts/python.exe scripts/prepare_fp8_calibration.py
.venv_gpu/Scripts/python.exe scripts/build_fp8_models.py
```

校正にはNYUv2の公式train795画像からseed43で選んだ64画像・58sceneを使い、GTは使いません。公式test654画像とはindexとsceneの重複がなく、3つの評価集合とのdecoded RGBの完全一致も確認します。前処理は評価と同じ518×518へのCUBIC直接リサイズです。`data_fp8_calibration/manifest.json`に選択ID、ハッシュ、重複確認を、`models_fp8/calibration.json`に各層の最大絶対値を保存します。

`models_fp8/fp8_w8a8.onnx`は、48個の定数重みMatMulの重み・入力をE4M3FNのFP8へ写したモデルです。重みはテンソル全体の最大絶対値、入力は校正64画像でのテンソル全体の最大絶対値からスケールを決めます。ほかの部分は既存FP16モデルの構成、外部の入出力はFP32を保ちます。`GemmFloat8`はFP8のA/BをcuBLASLtへ渡し、FP32 compute type・FP8 fast accumulation有効（`FAST_ACCUM=1`）・FP16出力で計算します。純粋なFP32累積という意味ではありません。

生成時に小さい行列と実際の1370トークンの行列で数値を照合し、全48演算のCUDA配置と有限出力を確認します。FP16基準には、同じopset21へ公式変換した`models_fp8/fp16_opset21.onnx`を使います。校正画像1枚で変換前のFP16との出力一致も確認し、各層のスケールとともに`models_fp8/metadata.json`へ記録します。

```powershell
.venv_gpu/Scripts/python.exe scripts/benchmark_fp8.py --fp16-model models_fp8/fp16_opset21.onnx --fp8-model models_fp8/fp8_w8a8.onnx --output-dir results_fp8/benchmark --rounds 3 --warmup 20 --iterations 100
.venv_gpu/Scripts/python.exe scripts/evaluate_dense.py --manifest data_dense/manifest.json --provider CUDA --output-dir results_fp8/nyu --models fp16=models_fp8/fp16_opset21.onnx fp8=models_fp8/fp8_w8a8.onnx --reference fp16 --no-previews
.venv_gpu/Scripts/python.exe scripts/evaluate_dense.py --manifest data_dense/sintel_manifest.json --provider CUDA --output-dir results_fp8/sintel --models fp16=models_fp8/fp16_opset21.onnx fp8=models_fp8/fp8_w8a8.onnx --reference fp16 --no-previews
```

速度はFP16→FP8→FP8→FP16とFP8→FP16→FP16→FP8を使って3巡し、各ブロックでウォームアップ20回・計測100回、TF32無効です。`results_fp8/benchmark/benchmark.json`の`status`が`complete`になったことを確認し、CPU上の入出力を使う時間と、同期を含むI/O Bindingの時間を比べます。モデル読込やprofile取得は計測外です。過去の測定がある出力先は上書きしないので、再測定時は別の`--output-dir`を指定します。

GT評価は`results_fp8/nyu/summary.json`と`results_fp8/sintel/summary.json`へ保存します。差の基準はFP16で、AbsRel・逆深度NRMSEの差が正なら悪化です。NYUv2は実測GTと補完GTを分け、Sintelは逆深度NRMSEを主にδ1とclip率も見ます。画像別のGT整合とscene/sequence単位の信頼区間は上のdense評価と同じです。

FP8演算がCUDA上で実行されたことと、モデル全体が速くなることは別です。このORT実装はGEMMごとに同期・一時領域確保・cuBLASLtハンドル生成を行います。保存したoptimized graphとprofileで型・配置を確認できますが、ネイティブTensor Core命令のtraceまでは行っていません。

## 保存した結果から図を作る

```powershell
.venv/Scripts/python.exe scripts/plot_evidence.py
.venv/Scripts/python.exe scripts/validate_evidence.py
.venv/Scripts/python.exe scripts/plot_fp8_evidence.py
.venv/Scripts/python.exe scripts/validate_fp8_evidence.py
```

`article_figures/`へ、公開した数値記録を使ったCPU/GPUグラフが出ます。元のRGB画像やGTファイルは含みません。検算スクリプトはCSVから正答数・誤差平均を再集計し、各回の推論時間から中央値を計算し直します。

NYUv2をGPUで128画像評価した後は、保存した予測から4bitの差が大きい画像も描けます。元RGBやGTの配列はこの図へ含めません。

```powershell
.venv/Scripts/python.exe scripts/plot_dense_example.py --results-dir results_dense_nyu_gpu --output-dir article_figures
```

## 全8構成の出力とAbsRelを揃える

FP32、FP16、dynamic INT8、INT4の2設定、W8A16、W4A16、FP8の値を同じ順で集計します。公開済みのCSVだけで表を作り直す場合は、次のコマンドを使います。モデル推論やGTファイルの取得は不要です。

```powershell
.venv/Scripts/python.exe scripts/summarize_all_models_dense.py --output-dir results/all-models
```

`all_models_metrics.csv`に、NYUv2の実測・補完GTとSintelの計24行が出ます。`all_models_summary.json`には元ファイルとモデルのハッシュ、FP32との差、撮影場所・シーケンス単位の95%信頼区間も記録します。既存の出力先は上書きしないので、再集計には別のフォルダーを指定してください。FP8節のFP16基準の差とは参照モデルが異なります。

出力画像も再生成する場合は、上の手順でNYUv2のGPU評価7構成とFP8評価を全128画像について済ませ、予測配列を残します。その後、次を実行します。

```powershell
.venv/Scripts/python.exe scripts/plot_all_models_dense.py --output-dir article_figures/all_models
```

固定例は選択リスト先頭の`nyu_0014`、悪化例はFP32以外の7構成×128画像で実測GTのAbsRel増加が最大だった`nyu_0861`です。2列4行のまとめ図2枚、モデル別16枚、画像ごとのAbsRelと選定・表示条件の記録を出力します。記事の図と記録は`article_figures/all_models/`にあります。

各例の全8出力の最小値・最大値を共通の色域とし、モデルごとの正規化や範囲外の切り捨ては行いません。図はGTへ整合する前の逆深度で、AbsRelは実測GTの共通有効画素で倍率・切片を合わせた後の値です。元RGBやGT画像は描画しません。画像のAbsRelは1枚ごとの値で、128画像平均とは別です。

## 途中で止まった場合

- `No module named depth_anything_v2`：上の公式repoを`vendor/Depth-Anything-V2`へ取得したか確認します。モデル重みだけでは実装は揃いません。
- DLLの読み込みに失敗する：指定したvenvのPythonとrequirementsを使っているか確認します。`runtime_init.py`はWindows用の同梱MSVC DLLを先に読み込みます。
- パッケージ導入で`WinError 3`と長いパスが出る：深い作業フォルダー内のvenvでWindowsのパス長制限へ当たった実例がありました。短い作業場所へcloneし直してvenvを作ります。既存環境やシステム設定を変更する前に、パスを短くして確認してください。
- CUDAがない、またはCUDA初期化エラー：CPU用とGPU用venvを取り違えていないか、Provider一覧とドライバーを確認します。実行ログだけでGPU計算と判断しません。
- `Checkpoint configuration differs`：同じ出力フォルダーに別条件を混ぜています。既存結果を残し、別の`--output-dir`を指定します。
- データ取得が止まった：既存ファイルのハッシュ・途中ファイルを確認します。データを別のsplitへ勝手に替えず、元のエラーを確認して再実行してください。

## 公式資料

- [Depth Anything V2公式実装](https://github.com/DepthAnything/Depth-Anything-V2)、[Small checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Small)、[論文](https://arxiv.org/abs/2406.09414)
- [ONNX Runtimeの量子化](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)、[CUDA Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ORT 1.31.0のGemmFloat8実装](https://github.com/microsoft/onnxruntime/blob/v1.31.0/onnxruntime/contrib_ops/cuda/math/gemm_float8.cu)、[cuBLASの低精度計算](https://docs.nvidia.com/cuda/cublas/index.html#narrow-precision-data-types-usage)
- [DA-2K](https://huggingface.co/datasets/depth-anything/DA-2K)、[NYUv2作者配布元](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)、[MPI Sintel depth](https://sintel.is.tue.mpg.de/depth)

Small checkpointと公式実装はApache-2.0です。NYUv2とMPI Sintelについて、確認した作者配布資料から明示的な再配布ライセンスは確認できていません。映画Sintel自体のCC BY 3.0とMPIの変更された評価データを同じものとして扱わず、元データはそれぞれの公式配布元から取得してください。
