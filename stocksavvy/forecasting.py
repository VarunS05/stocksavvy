"""Price forecasting with an honest evaluation methodology.

Two things the original version of this project got wrong, fixed here:
1. It evaluated with a *random* train/test split on time-series data, which
   leaks future information into training (the model can effectively see
   "the future" during training via nearby shuffled rows). This version uses
   a chronological split -- train on the earlier period, test on the later
   period, like a real forecasting deployment would.
2. It computed a "sentiment-adjusted" prediction, but the code path that
   actually fed live news sentiment into the model was never wired up (dead,
   commented-out code) -- the app silently ignored whatever news text you
   typed. This version scores the input text with VADER and uses it as a
   real input feature, and reports accuracy with vs. without sentiment so
   you can see whether it's actually helping.
3. It predicted the raw closing price level. On a chronological split this
   fails badly for a trending series: a tree-based model can't extrapolate
   past the price range it saw in training, and the test period here sits
   well above anything in the training period (this was verified directly --
   predicting price level gave R2 around -2, i.e. much worse than just
   guessing the mean). Predicting the forward *return* instead keeps the
   target roughly stationary regardless of the underlying trend, which is
   the standard fix for this failure mode.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .indicators import add_technical_indicators

PRICE_FEATURES = ["Close", "bb_high", "bb_low", "macd", "rsi", "sma", "ema"]
SENTIMENT_FEATURES = ["subjectivity", "polarity", "compound", "neg", "neu", "pos"]


def prepare_dataset(df, horizon=5, use_sentiment=True):
    """Adds indicators, builds the feature matrix, and a target column that is
    the *forward return* (not raw price) `horizon` trading days ahead of each row:
    (Close[t+horizon] - Close[t]) / Close[t].
    """
    df = add_technical_indicators(df)
    feature_cols = PRICE_FEATURES + (SENTIMENT_FEATURES if use_sentiment else [])
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    df["target"] = (df["Close"].shift(-horizon) - df["Close"]) / df["Close"]
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    return df, feature_cols


def chronological_split(df, test_frac=0.2):
    n_test = max(1, int(len(df) * test_frac))
    return df.iloc[:-n_test], df.iloc[-n_test:]


def train_and_evaluate(df, horizon=5, model_factory=None):
    """Trains price-only and price+sentiment models with a chronological
    holdout, and returns their test-set R2/MAE side by side -- alongside a
    "predict zero return" (random-walk) baseline, which is the real
    benchmark for short-horizon return forecasting, not just R2's implicit
    comparison to the test set's own mean.
    """
    if model_factory is None:
        model_factory = lambda: RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)

    results = {}
    for use_sentiment in (False, True):
        prepared, feature_cols = prepare_dataset(df, horizon, use_sentiment)
        train_df, test_df = chronological_split(prepared)

        scaler = StandardScaler().fit(train_df[feature_cols].values)
        X_train = scaler.transform(train_df[feature_cols].values)
        X_test = scaler.transform(test_df[feature_cols].values)

        model = model_factory()
        model.fit(X_train, train_df["target"].values)
        preds = model.predict(X_test)
        y_test = test_df["target"].values

        key = "with_sentiment" if use_sentiment else "price_only"
        results[key] = {
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "random_walk_baseline_mae": float(np.mean(np.abs(y_test))),
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
        }
    return results


def forecast_next_close(df, model, scaler, feature_cols, live_sentiment=None):
    """Predicts the closing price `horizon` trading days out from the most
    recent row of `df`, by predicting the forward return and applying it to
    the latest known close (the model itself is trained on returns, not
    price levels -- see prepare_dataset). If `live_sentiment` (a dict from
    sentiment.score_text) is given, it overrides that row's sentiment
    features -- this is how a live news headline actually reaches the
    prediction, unlike the original app.
    """
    enriched = add_technical_indicators(df)
    latest = enriched.iloc[[-1]].copy()
    latest_close = float(latest["Close"].iloc[0])

    if live_sentiment is not None:
        latest["compound"] = live_sentiment["compound"]
        latest["neg"] = live_sentiment["neg"]
        latest["neu"] = live_sentiment["neu"]
        latest["pos"] = live_sentiment["pos"]
        if "polarity" in feature_cols:
            latest["polarity"] = live_sentiment["compound"]
        if "subjectivity" in feature_cols:
            latest["subjectivity"] = 1 - live_sentiment["neu"]

    x = scaler.transform(latest[feature_cols].values)
    predicted_return = float(model.predict(x)[0])
    return latest_close * (1 + predicted_return)
