# keiba-ai-app

競馬AIの予想、買い目提案、期間シミュレーション、レース情報更新をWeb画面から使えるアプリです。

## 現在の推奨モデル

通常の予想・買い目提案では、総合推奨モデルとして **`no_market_lambdarank_v3_g40_8_3_1`** を使います。

```text
1着 = 40
2着 = 8
3着 = 3
4着 = 1
5着以下 = 0
```

このモデルは `no_market_lambdarank_v2` をベースに、1着の強さを保ちながら2〜4着にも報酬を与え、通常買い目の回収率を改善する目的で作ったLambdaRankモデルです。

Kelly AIだけは、2025年検証で最も回収率が高かった **`no_market_lambdarank_v2`** を専用で使います。

| 用途 | 使用モデル | 理由 |
|---|---|---|
| 通常予想・回収率重視 | `no_market_lambdarank_v3_g40_8_3_1` | `roi_focus` の2025年回収率が最良 |
| 的中率重視 | `no_market_lambdarank_v3_g40_8_3_1` | `hit_focus` でもプラス回収 |
| Kelly AI | `no_market_lambdarank_v2` | Kellyシミュレーション回収率が最良 |
| 比較用 | `market` | オッズ・人気ありの従来モデル |

## 2025年ホールドアウト検証

学習データは2017〜2024年、検証データは2025年です。以下は2025年通年、1レース予算3,000円、Kelly資金100,000円でのシミュレーション結果です。

### 予測精度

| モデル | 重み | 単勝率 | 複勝率 | 予想上位3頭の3着内率 | WIN AUC |
|---|---|---:|---:|---:|---:|
| `no_market_lambdarank_v2` | 1着40 / 2〜3着1 | **44.49%** | 66.86% | 48.44% | **0.8735** |
| `no_market_lambdarank_v3_g40_8_3_1` | 40 / 8 / 3 / 1 | 42.66% | 66.57% | **50.04%** | 0.8654 |
| `no_market_lambdarank_v3_g40_5_2_1` | 40 / 5 / 2 / 1 | 43.88% | **67.00%** | 49.82% | 0.8686 |

### 回収率

| モデル | スタイル | 回収率 | 的中率 | 購入R | 収支 |
|---|---|---:|---:|---:|---:|
| `no_market_lambdarank_v3_g40_8_3_1` | `roi_focus` | **106.66%** | 55.40% | 3,451 | +689,931円 |
| `no_market_lambdarank_v3_g40_5_2_1` | `roi_focus` | 106.22% | 55.36% | 3,450 | +644,040円 |
| `no_market_lambdarank_v2` | `roi_focus` | 103.69% | 54.31% | 3,449 | +381,459円 |
| `no_market_lambdarank_v3_g50_10_4_1` | `hit_focus` | **102.46%** | 88.60% | 3,455 | +254,481円 |
| `no_market_lambdarank_v3_g40_8_3_1` | `hit_focus` | 102.32% | 88.36% | 3,455 | +240,929円 |
| `no_market_lambdarank_v2` | `kelly_ai` | **190.39%** | 11.10% | 2,974 | +5,174,462円 |
| `no_market_lambdarank_v3_g40_5_2_1` | `kelly_ai` | 185.71% | 11.30% | 2,973 | +4,923,924円 |
| `no_market_lambdarank_v3_g40_8_3_1` | `kelly_ai` | 182.17% | 11.64% | 2,980 | +4,736,867円 |

## 主なファイル

| ファイル | 役割 |
|---|---|
| `src/race_scraper.py` | netkeiba/JRAオッズから出走表、結果、払戻、オッズを取得 |
| `src/preprocess.py` | 過去データを学習用DataFrameに整形 |
| `src/train_no_market_lambdarank_v2.py` | 既存Kelly向けLambdaRank v2を学習 |
| `src/train_no_market_lambdarank_v3_top4.py` | 2〜4着にも重みを与えるv3系LambdaRankを学習 |
| `src/predictor.py` | 学習済みモデルを読み込み、各馬のスコア・順位・確率を計算 |
| `src/betting.py` | `roi_focus`、`hit_focus` などの買い目を生成 |
| `src/kelly_betting.py` | Kelly AI用の馬連・馬単確率と資金配分を計算 |
| `src/web_app.py` | Flask Webアプリ本体 |
| `src/templates/index.html` | Web画面 |
| `tools/compare_lambdarank_roi.py` | LambdaRankモデルごとの回収率比較 |

## 使い方

```powershell
python -X utf8 src\web_app.py --host 127.0.0.1 --port 7860
```

ブラウザで開きます。

```text
http://127.0.0.1:7860/
```

ローカル作業用パッケージでは、次のように起動します。

```powershell
cd C:\Users\yukim\Downloads\keiba_ai_package\keiba_ai
python -X utf8 src\web_app.py --host 127.0.0.1 --port 7860
```

## 新しいLambdaRank重みの学習

`src/train_no_market_lambdarank_v3_top4.py` は環境変数でモデル名と `label_gain` を変えられます。

```powershell
$env:KEIBA_MODEL_VERSION='no_market_lambdarank_v3_g40_8_3_1'
$env:KEIBA_LABEL_GAIN='0,1,3,8,40'
python -X utf8 src\train_no_market_lambdarank_v3_top4.py
```

`label_gain` はLightGBMのラベル順なので、上の例は次の意味です。

```text
5着以下 = 0
4着 = 1
3着 = 3
2着 = 8
1着 = 40
```

## 回収率比較

```powershell
python -X utf8 tools\compare_lambdarank_roi.py --start 2025-01-01 --end 2025-12-31 --budget 3000 --bankroll 100000
```

結果は次に保存されます。

```text
data/roi_compare_lambdarank_models.json
```

## テスト

```powershell
python -X utf8 tools\run_tests.py --fail-under 80
```

## 公開しないデータ

スクレイピングで取得したレースデータ、払戻、オッズ、学習済みモデル、ログはGitHubに公開しない前提です。

```text
data/
models/
logs/
*.pkl
*.pkl.gz
*.csv
```

## 注意

このアプリの予想、勝率、自信度、買い目、回収率は参考値です。実際の馬券購入は自己責任で行ってください。
