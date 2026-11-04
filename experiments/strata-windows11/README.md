# Windows 11でStrataを動かす

StrataのOpenAI互換APIへ日本語の説明文とPythonの関数生成を依頼し，最初の本文が届くまでの時間と生成速度を記録するコードです．モデルやエンジンは同梱していません．

導入手順と検証結果の記事：[StrataをWindows 11で試す。RTX 5070 Ti 16GBとメモリ32GBで動かした](https://yuyuyuroom.site/2026/11/04/strata-windows11-rtx5070ti-32gb/)

Strataの導入は[公式のv0.2.10リリース](https://github.com/Niko1221/Strata/releases/tag/v0.2.10)と[Terminalの説明](https://github.com/Niko1221/Strata/blob/v0.2.10/docs/TERMINAL_LAUNCH.md)を参照してください．このコード自体はPython標準ライブラリだけを使います．

## 確認した構成と結果

2026年11月4日に，Windows 11 Home 26H2，Core i7-14700F，RAM 32GB，RTX 5070 Ti 16GB，NVIDIAドライバー610.88，NVMe SSDで実行しました．Strata 0.2.10，ControlとCUDAエンジン0.1.56，配布物に同梱のPython 3.12.15です．

モデルはQwen3.8-Flash-Next ISTA-DASLab Q2_0，コンテキスト8,192，並列1，画像なし，DFlash2，VRAM予備1,152MiBです．低メモリのmmapモードで，`STRATA_ROUTE_TAIL_SKIP=7`は既定のままです．初回準備で作るエキスパートファイルを含め，モデル関連フォルダーは約102.8GBでした．

初回の補助ファイル展開は262文字のパスで失敗し，`STRATA_INSTALL_DIR`に短い準備用フォルダーを指定して再開しました．OSの設定やStrataのコードは変更していません．

| 依頼 | 本文開始の中央値 | 完了までの中央値 | 生成速度の中央値 |
| --- | ---: | ---: | ---: |
| VRAMとRAMの説明 | 0.466秒 | 4.58秒 | 54.5tokens/s |
| Python関数の生成 | 0.771秒 | 1.52秒 | 50.2tokens/s |
| MoEの説明 | 0.722秒 | 3.78秒 | 75.6tokens/s |

表はウォームアップ後の3回です．最初のVRAM説明は本文開始3.62秒，完了23.83秒，生成速度13.2tokens/sでした．キャッシュの効果を含みます．Pythonの短い38トークンの出力も，生成速度は14.1→50.2→81.3tokens/sと変化しました．

日本語の説明には中国語の語句が3回とも混ざりました．通常の推論設定で追加した1回も同様です．生成された短いXOR関数は3回とも同じ実装で，内容を確認してからそれぞれ106入力で検査し，計318回で期待値と一致しました．モデルや実用コード全般の品質評価ではありません．

## 実行

Strataをローカルで起動し，`http://127.0.0.1:8080/health`と`/v1/models`で準備完了を確認してから，このフォルダーで実行します．

```powershell
python measure_strata.py --output results/strata-results.json
```

システム側にPythonがない場合は，StrataのControl配布物にある`python/python.exe`を使えます．たとえばこのフォルダーの`strata-runtime`へ展開した場合は，次のように実行します．

```powershell
& .\strata-runtime\python\python.exe .\measure_strata.py --output .\results\strata-results.json
```

認証を有効にしたサーバーでは，環境変数`STRATA_API_KEY`へ実際のキーを設定します．キーは結果JSONへ書き出しません．接続先を変更する場合は`--base-url`を指定します．

最初に1回ウォームアップし，続いて3種類の依頼を順番に3回ずつ実行します．条件は`temperature=0`，`seed=42`，`enable_thinking=false`です．並列リクエストは送りません．`max_tokens`は依頼ごとに256～640です．`finish_reason=length`の回答は上限で途中終了しています．

結果の`first_content_seconds`はHTTPリクエスト開始から最初の空でない本文差分まで，`elapsed_seconds`はストリーム終了までです．`timings.predicted_per_second`はサーバーが報告した生成部分の速度で，入力処理時間を含みません．`timings.cache_n`と`prompt_n`も残すので，入力キャッシュの有無を確認できます．tokens/sを文字数/秒と読み替えないでください．

同じプロンプトを繰り返すため，エキスパートや入力のキャッシュが効く条件です．コールド起動や長文処理を代表するベンチマークではありません．別のモデルや別の入力条件で測った速度と，エンジンの優劣を直接比較するものでもありません．

生成した回答は`answer`に保存します．回答内容を自動実行する処理はありません．モデルの性能全般や実用コードの正しさを，この3問だけで保証するものではありません．

結果JSON，取得したモデル，配布ZIP，生ログはGitの対象から除外しています．
