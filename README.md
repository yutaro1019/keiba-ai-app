# keiba-ai-app

競馬AIの予想・買い目提案・シミュレーション・情報更新をWeb画面から使えるアプリです。

## モデル・予想精度

### LambdaRank v2（推奨モデル）

LambdaRankによる直接順位最適化モデル。学習データ2017〜2024年、検証データ2025年。

| 指標 | 数値 |
|---|---:|
| 学習データ | 2017〜2024年 |
| 検証データ | 2025年（未使用ホールドアウト）|
| 予想1位の単勝的中率 | **44.5%** |
| 予想1位の複勝的中率 | **66.9%** |
| WIN AUC | **0.8735** |
| 特徴量数 | 50（重要度上位） |
| モデル数（アンサンブル） | 3シード |

### 旧バイナリモデル（market / no_market系）

| 指標 | market | no_market v10 |
|---|---:|---:|
| 予想1位の単勝的中率 | 33.2% | 33.1% |
| 予想1位の複勝的中率 | 63.2% | 62.9% |
| WIN AUC | 0.8370 | 0.8246 |

## Kelly基準 資金分配AI（バックテスト済）

LambdaRank v2 の予想スコアとマーケットオッズを比較し、期待値プラスの馬券だけをKelly基準で選択します。

### バックテスト結果（2025年 3,455レース）

| 券種 | ベット数 | 的中率 | ROI |
|---|---:|---:|---:|
| 馬連（umaren） | 9,428 | 3.5% | **+39.5%** |
| 馬単（umatan） | 18,987 | 1.8% | **+46.1%** |
| **合計** | 28,415 | 2.4% | **+43.9%** |

- 初期資金100,000円 → 最終12,563,895円（フラット1,000円/ベット）
- 複勝・三連複は除外推奨（ROI マイナス）

### 推奨設定

| パラメータ | 値 | 説明 |
|---|---|---|
| 対象券種 | 馬連・馬単 | バックテストで正ROIを確認 |
| min_ev | 0.10 | マーケット比10%以上の優位性のみ |
| top_k | 4 | 上位4頭の組合せを対象 |
| kelly_factor | 0.25 | クォーターKelly（分散低減） |

### 仕組み

1. **モデル確率**: LambdaRank v2スコア → Plackett-Luce softmax
2. **マーケット確率**: 単勝オッズから逆算（no_market特徴量と独立）
3. **Kelly係数**: `f* = (モデル確率 × マーケットオッズ - 1) / (オッズ - 1)`
4. **組合せオッズ推定**: 単勝オッズ + Plackett-Luceで全券種のオッズを導出

## モデル切り替え

| モデル | 説明 | 推奨用途 |
|---|---|---|
| `no_market_lambdarank_v2` | LambdaRank v2（**推奨**） | Kelly資金分配との組み合わせ |
| `market` | 通常モデル（オッズ・人気あり） | 単純な人気馬選び |
| `no_market_v10` | バイナリモデル v10 | 旧来互換 |

CLIでは `--model-variant no_market_lambdarank_v2`、Webアプリでは「予想モデル」から切り替えます。

## 使い方

### Webアプリ起動

```powershell
cd C:\Users\yukim\Downloads\keiba_ai_package\keiba_ai
python -X utf8 src\web_app.py --host 127.0.0.1 --port 7860
```

ブラウザで `http://127.0.0.1:7860` を開きます。

### Kelly資金分配バックテスト

```powershell
cd keiba_ai
python src\backtest_kelly.py --years 2025 --kelly 0.25 --min-ev 0.1 --top-k 4 --tickets umaren umatan --flat-bet 1000
```

### バックテストオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--years` | 2025 | 評価年（2025のみ推奨: 学習外データ）|
| `--kelly` | 0.25 | Kelly係数の倍率 |
| `--min-ev` | 0.0 | 最低期待値フィルタ（0.1推奨）|
| `--top-k` | 5 | 組合せ馬券の対象上位頭数（4推奨）|
| `--tickets` | 全券種 | 対象券種 |
| `--flat-bet` | 0 | 固定額ベット（0=Kelly複利）|

### Python API（predictor連携）

```python
from predictor import KeibaPredictor
from kelly_betting import load_odds_live

predictor = KeibaPredictor(model_variant="no_market_lambdarank_v2")
race_df = ...  # 出走馬データ
odds_data = load_odds_live(race_id, data_dir="data/")

# Kelly推薦馬券を取得
bets = predictor.recommend_bets(
    race_df,
    odds_data,
    bankroll=100_000,
    kelly_factor=0.25,
    min_ev=0.1,
    top_k=4,
    ticket_types=["umaren", "umatan"],
)

for b in bets:
    print(f"{b['ticket_kind']} {b['horses']}  "
          f"EV={b['ev']:+.1%}  掛け金={b['bet_amount']:,}円")
```

### CLIバックテスト（バイナリモデル旧来）

```powershell
python -X utf8 src\keiba_ai.py --no-banner --backtest --year 2025 --budget 3000 --style hybrid --model-variant market
```

## 主な機能

- 未来レース予想（LambdaRank v2 / market モデル対応）
- Kelly基準による馬連・馬単の資金配分推薦（ROI +43.9% 実績）
- 日付指定で全レース予想
- 競馬場とR指定で個別レース予想
- 予想順位・自信度・推定勝率・推定3着内率の表示
- 期間指定シミュレーション
- 情報更新モード（スクレイピング＋lookupテーブル更新）
- 実払い戻しデータを使った検証

## 情報更新とデータについて

スクレイピングで取得したデータは、netkeiba等の利用規約に配慮するため、このリポジトリでは公開しません。必要な場合は各自の環境で情報更新モードを実行して取得してください。

情報更新モードでは、新しく追加されたレース結果を学習用データと同じ形式に整え、LambdaRank v2モデルのlookupテーブルも更新します。

## 注意

このアプリの予想、勝率、自信度、買い目、回収率は参考値です。実際の馬券購入は自己責任で行ってください。
