"""Technical indicators on OHLCV price data, via the `ta` library."""
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import BollingerBands


def add_technical_indicators(df, close_col="Close"):
    """Adds Bollinger Bands, MACD, RSI, SMA, EMA columns to a copy of df."""
    df = df.copy()
    close = df[close_col]

    bb = BollingerBands(close)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["macd"] = MACD(close).macd()
    df["rsi"] = RSIIndicator(close).rsi()
    df["sma"] = SMAIndicator(close, window=14).sma_indicator()
    df["ema"] = EMAIndicator(close).ema_indicator()

    return df
