# NV-Tesseractの実験

NVIDIAのSDKで予測・異常検知を動かすスクリプトと、国内株5銘柄＋USD/JPYの評価コードです。SDKとチェックポイントは別途取得します。

WSL 2、NVIDIA GPU、CUDA対応ドライバー、Git、uvが必要です。記事のSDKコミットは `b68ea6f946b9b299d48f91673fd722366c150e99`。forecastingはPython 3.12.14、PyTorch 2.12.1+cu130で実行しました。モデルを含めて10 GB以上の空きを見込んでください。

このリポジトリのルートから実行します。

```bash
git clone https://github.com/NVIDIA/NV-Tesseract.git nv_tesseract_lab/NV-Tesseract
git -C nv_tesseract_lab/NV-Tesseract checkout b68ea6f946b9b299d48f91673fd722366c150e99
cd nv_tesseract_lab/NV-Tesseract/forecasting
uv sync --python 3.12
.venv/bin/python ../../run_forecasting_probe.py
```

初回はチェックポイントをダウンロードします。続けて国内株と為替を試す場合も、カレントディレクトリは `forecasting` のままです。

```bash
.venv/bin/python ../../run_japan_stocks_fx_backtest.py
.venv/bin/python ../../rebuild_japan_stocks_fx_aggregates.py
.venv/bin/python ../../validate_japan_stocks_fx_backtest.py
.venv/bin/python ../../run_toyota_learning_regime_experiment.py
.venv/bin/python ../../analyze_toyota_learning_regime.py
```

結果は `nv_tesseract_lab/results/`。5銘柄の予測を先に実行してから集計・検算します。トヨタの追加学習は6回分の学習を含みます。

相場データは実行時にYahoo Financeから取得します。末尾の評価期間が移動し、株式分割などによって過去データも改訂されるため、記事の数値と一致する保証はありません。取得したデータ、分割、予測配列、指標を一緒に残してください。このリポジトリには相場データやNVIDIAのモデル重みは含めていません。利用条件は各配布元で確認してください。

異常検知は別の仮想環境を使います。

```bash
cd ../ad_diffusion
uv sync --python 3.12
.venv/bin/python examples/quick_example.py --download-weights
.venv/bin/python ../../run_ad_probe.py
.venv/bin/python ../../run_ad_seeded_repeat.py
```

最後のスクリプトは乱数シードを固定した比較です。最初の動作確認と同じ精度が出るとは限らないため、結果は別々に保存します。学習・推論の操作は [NVIDIAのREADME](https://github.com/NVIDIA/NV-Tesseract) も参照してください。
