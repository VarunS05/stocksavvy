"""StockSavvy: sentiment-aware stock price forecasting, in Streamlit.

Modes:
- Visualize / Recent Data: technical indicators on live-downloaded price data (yfinance).
- Forecast: trains price-only and price+sentiment models on the historical
  dataset with a chronological train/test split, and reports both so you can
  see whether sentiment actually helps for the chosen horizon.
- Forecast with news: same as above, but scores whatever headline you type
  with VADER and feeds it into the prediction for the most recent day.
"""
import datetime

import streamlit as st
import yfinance as yf
import pandas as pd

from stocksavvy import add_technical_indicators, score_text, train_and_evaluate, forecast_next_close

HISTORY_PATH = "data/stock_sentiment_history.csv"


@st.cache_data
def load_history():
    return pd.read_csv(HISTORY_PATH, parse_dates=["Date"])


@st.cache_data
def download_price_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, progress=False)


def page_visualize():
    st.header("Technical Indicators")
    ticker = st.sidebar.text_input("Ticker", value="SPY").upper()
    duration = st.sidebar.number_input("Days of history", value=500, min_value=30)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=duration)
    data = download_price_data(ticker, start_date, end_date)

    if data.empty:
        st.error(f"No data returned for '{ticker}'.")
        return

    indicator = st.radio("Indicator", ["Close", "Bollinger Bands", "MACD", "RSI", "SMA", "EMA"])
    enriched = add_technical_indicators(data)
    columns = {
        "Close": ["Close"],
        "Bollinger Bands": ["Close", "bb_high", "bb_low"],
        "MACD": ["macd"],
        "RSI": ["rsi"],
        "SMA": ["sma"],
        "EMA": ["ema"],
    }[indicator]
    st.line_chart(enriched[columns])
    st.subheader("Recent data")
    st.dataframe(data.tail(10))


def page_forecast(with_news):
    st.header("Forecast" + (" with news" if with_news else ""))
    history = load_history()
    horizon = st.number_input("Forecast horizon (trading days ahead)", value=5, min_value=1, max_value=30)

    news_text = ""
    if with_news:
        news_text = st.text_area("Today's news headline / summary")

    if st.button("Train & forecast"):
        with st.spinner("Training price-only and price+sentiment models (chronological holdout)..."):
            results = train_and_evaluate(history, horizon=horizon)

        st.subheader("Model comparison (chronological holdout, not shuffled)")
        st.caption("MAE is on forward *returns*, not price. Random-walk baseline = always predicting 0% change.")
        for key, label in [("price_only", "Price + indicators only"), ("with_sentiment", "Price + indicators + sentiment")]:
            r = results[key]
            st.write(
                f"**{label}** — R²: {r['r2']:.3f}, MAE: {r['mae']:.4f} "
                f"(random-walk baseline MAE: {r['random_walk_baseline_mae']:.4f})"
            )

        live_sentiment = None
        if with_news and news_text.strip():
            live_sentiment = score_text(news_text)
            st.write("VADER sentiment for your headline:", live_sentiment)

        target = results["with_sentiment"] if with_news else results["price_only"]
        prediction = forecast_next_close(
            history, target["model"], target["scaler"], target["feature_cols"], live_sentiment=live_sentiment
        )
        st.success(f"Predicted close, {horizon} trading days out: {prediction:.2f}")
        st.caption(
            "R² compares the two models on identical held-out data; a higher with-sentiment R² means "
            "the headline you typed genuinely moved the prediction, not just cosmetically."
        )


def main():
    st.title("StockSavvy")
    page = st.sidebar.selectbox("Mode", ["Visualize", "Forecast", "Forecast with news"])
    if page == "Visualize":
        page_visualize()
    elif page == "Forecast":
        page_forecast(with_news=False)
    else:
        page_forecast(with_news=True)


if __name__ == "__main__":
    main()
