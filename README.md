# StockSavvy

Sentiment-aware stock price forecasting: combines price/technical indicators with news-sentiment scoring (VADER) to forecast returns, and reports honest, chronologically-validated results — including comparisons against a naive baseline, so a headline number is never presented without something to check it against.

## Why this version is different from the original

The original version of this project had three real problems, all fixed here:

1. **The news-sentiment feature was never actually used.** The UI let you type a headline, but the code path that would score it and feed it into the prediction was commented out — the "Predict with news" mode silently ignored your input and predicted from a static file instead. This version scores your text with VADER and actually uses it as a model feature.
2. **Evaluation used a random train/test split on time-series data** — which leaks future rows into training. This version uses a **chronological split**: train on the earlier period, test only on the later period, like a real deployment.
3. **It predicted raw price level, not return.** On a chronologically-split trending series, this fails outright — a tree-based model can't extrapolate past the price range it saw in training (verified directly: predicting price level gave R² around **-2.0**, i.e. far worse than useless). Predicting the forward *return* instead is the standard fix, and is what this version does.

## Results

```
python -c "
import pandas as pd
from stocksavvy import train_and_evaluate
df = pd.read_csv('data/stock_sentiment_history.csv', parse_dates=['Date'])
print(train_and_evaluate(df, horizon=5))
"
```

| Horizon | Model | R² | MAE (return) | Random-walk baseline MAE |
|---|---|---|---|---|
| 1 day | Price + indicators only | -0.310 | 0.0165 | 0.0140 |
| 1 day | + sentiment | -0.138 | 0.0150 | 0.0140 |
| 5 days | Price + indicators only | -0.477 | 0.0405 | 0.0305 |
| 5 days | + sentiment | -0.373 | 0.0386 | 0.0305 |

**Read honestly**: neither model beats the random-walk ("predict no change") baseline at either horizon — consistent with the well-documented difficulty of short-horizon return forecasting on liquid markets. What *is* a real, reproducible signal: adding sentiment features **consistently improves both R² and MAE over price-only**, at both horizons tested. That's the actual finding this project supports — sentiment helps at the margin, it doesn't produce a market-beating model, and the README says so rather than reporting only the flattering number.

## App

```bash
pip install -r requirements.txt
streamlit run app.py
```

- **Visualize** — live price data (via `yfinance`) with Bollinger Bands, MACD, RSI, SMA, EMA.
- **Forecast** — trains price-only and price+sentiment models on the historical dataset, chronological holdout, reports both plus the random-walk baseline.
- **Forecast with news** — same, but scores whatever headline you type with VADER and uses it as a live input feature for the most recent day's prediction.

## Project structure

```
stocksavvy/
  indicators.py    # Bollinger Bands, MACD, RSI, SMA, EMA (via `ta`)
  sentiment.py      # VADER news-sentiment scoring
  forecasting.py     # dataset prep (return target), chronological CV, evaluation with baseline
app.py                # Streamlit UI
data/
  stock_sentiment_history.csv  # historical OHLCV + sentiment features, 2014-2021
```

## Limitations

- Historical dataset is a single instrument's daily OHLCV + pre-computed sentiment features (2014–2021); results won't necessarily transfer to other tickers or regimes.
- Random Forest on hand-crafted features, not a model built for financial time series specifically (e.g. an ARIMA/GARCH volatility model or a proper sequence model would be a natural next step).
- VADER is a general-purpose lexicon-based sentiment tool, not fine-tuned on financial text — a finance-specific model (e.g. FinBERT) would likely score financial headlines more accurately.
- "Random-walk baseline" here means predicting zero return, the standard naive benchmark for short-horizon forecasting — it is not a real trading strategy (no transaction costs, slippage, or risk are modeled anywhere in this project).
