## PyTorch eval / no_grad / inference_mode・Ollama num_predict・Google Colab 200 CCUの検証

### 関連公開記事

- [PyTorchのmodel.eval()だけでは勾配計算は止まらない。no_gradとinference_modeの違いを確認する](https://yuyuyuroom.site/2026/11/12/pytorch-eval-no-grad-inference-mode/)
- [Ollamaの回答が途中で切れるときに確認するnum_predictとdone_reason。上限16・64・256トークンで実測する](https://yuyuyuroom.site/2026/11/12/ollama-num-predict-done-reason-length/)
- [Google AI ProのColab特典200 CCUでGPUは何時間使えるか。T4・L4・A100の表示消費率を調べた](https://yuyuyuroom.site/2026/11/12/google-ai-pro-colab-200-ccu-gpu-hours/)

### スクリプト・Colabノートブック・測定データ

- `ai_lab/experiment_eval_grad_modes.py`
  - 測定結果: `evidence/eval-grad-modes/cpu.json`, `evidence/eval-grad-modes/cuda.json`, `evidence/eval-grad-modes/colab.json`
  - Colabノートブック: [`notebooks/eval_grad_modes_colab.ipynb` (Open in Colab)](https://colab.research.google.com/github/matrizea/yuyuyuroom-ai-experiments/blob/278a1d2f2e4e4d818f18b088de1fb420ccc77660/notebooks/eval_grad_modes_colab.ipynb)
- `ai_lab/experiment_ollama_output_limit.py`
  - 測定結果: `evidence/ollama-output-limit/result.json`
- Colabランタイム表示観測データ: `evidence/colab-runtime-options/observations.json`

### 最小実行コマンド

```bash
git clone https://github.com/matrizea/yuyuyuroom-ai-experiments.git
cd yuyuyuroom-ai-experiments

# 実験1: PyTorch model.train()/eval() と no_grad / inference_mode の確認
python ai_lab/experiment_eval_grad_modes.py --device cpu --output ai_lab/results/eval_grad_cpu.json
python ai_lab/experiment_eval_grad_modes.py --device cuda --output ai_lab/results/eval_grad_cuda.json

# 実験2: Ollama num_predict (16 / 64 / 256) と done_reason の確認
read -r -p 'Ollamaサーバーの接続先URL: ' OLLAMA_HOST
export OLLAMA_HOST
python ai_lab/experiment_ollama_output_limit.py --output ai_lab/results/ollama_output_limit.json
```
