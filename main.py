import numpy as np
import pandas as pd
import yfinance as yf 
import warnings

warnings.simplefilter("ignore")

print("Descargando datos...")

START = "2010-01-01"

# =====================================================
# DESCARGAS
# =====================================================

tickers = {
    "gold": "GC=F",
    "dxy": "DX-Y.NYB",
    "tnx": "^TNX",
    "vix": "^VIX",
    "silver": "SI=F",
    "oil": "CL=F",
    "sp500": "^GSPC",
    "gdx": "GDX",
    "gdxj": "GDXJ"
}

data = {}

for name, ticker in tickers.items():
    print(f"Descargando {name}...")

    df = yf.download(
        ticker,
        start=START,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data[name] = df

gold = data["gold"]

dataset = pd.DataFrame(index=gold.index)

# =====================================================
# PRECIO
# =====================================================

dataset["oro_price"] = gold["Close"]

# =====================================================
# VARIABLES EXTERNAS
# =====================================================

dataset["dxy"] = data["dxy"]["Close"]
dataset["tnx"] = data["tnx"]["Close"]
dataset["vix"] = data["vix"]["Close"]
dataset["silver"] = data["silver"]["Close"]
dataset["oil"] = data["oil"]["Close"]
dataset["sp500"] = data["sp500"]["Close"]
dataset["gdx"] = data["gdx"]["Close"]
dataset["gdxj"] = data["gdxj"]["Close"]

# =====================================================
# RETORNOS ORO
# =====================================================

dataset["ret_1d"] = gold["Close"].pct_change(1)
dataset["ret_3d"] = gold["Close"].pct_change(3)
dataset["ret_5d"] = gold["Close"].pct_change(5)
dataset["ret_10d"] = gold["Close"].pct_change(10)
dataset["ret_20d"] = gold["Close"].pct_change(20)
dataset["ret_50d"] = gold["Close"].pct_change(50)


# =====================================================
# RETORNOS MACRO
# =====================================================

dataset["dxy_ret5"] = dataset["dxy"].pct_change(5)
dataset["tnx_ret5"] = dataset["tnx"].pct_change(5)
dataset["vix_ret5"] = dataset["vix"].pct_change(5)
dataset["silver_ret5"] = dataset["silver"].pct_change(5)
dataset["oil_ret5"] = dataset["oil"].pct_change(5)
dataset["sp500_ret5"] = dataset["sp500"].pct_change(5)
dataset["gdx_ret5"] = dataset["gdx"].pct_change(5)
dataset["gdxj_ret5"] = dataset["gdxj"].pct_change(5)

# =====================================================
# MEDIAS
# =====================================================

dataset["ma10"] = gold["Close"].rolling(10).mean()
dataset["ma20"] = gold["Close"].rolling(20).mean()
dataset["ma50"] = gold["Close"].rolling(50).mean()
dataset["ma100"] = gold["Close"].rolling(100).mean()
dataset["ma200"] = gold["Close"].rolling(200).mean()

# =====================================================
# DISTANCIAS
# =====================================================

dataset["dist_ma10"] = gold["Close"] / dataset["ma10"] - 1
dataset["dist_ma20"] = gold["Close"] / dataset["ma20"] - 1
dataset["dist_ma50"] = gold["Close"] / dataset["ma50"] - 1
dataset["dist_ma100"] = gold["Close"] / dataset["ma100"] - 1
dataset["dist_ma200"] = gold["Close"] / dataset["ma200"] - 1

# =====================================================
# VOLATILIDAD
# =====================================================

dataset["vol_5d"] = dataset["ret_1d"].rolling(5).std()
dataset["vol_20d"] = dataset["ret_1d"].rolling(20).std()
dataset["vol_50d"] = dataset["ret_1d"].rolling(50).std()

# =====================================================
# RSI
# =====================================================

delta = gold["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

dataset["rsi"] = 100 - (100 / (1 + rs))

# =====================================================
# MACD
# =====================================================

ema12 = gold["Close"].ewm(span=12, adjust=False).mean()
ema26 = gold["Close"].ewm(span=26, adjust=False).mean()

dataset["macd"] = ema12 - ema26

dataset["macd_signal"] = (
    dataset["macd"]
    .ewm(span=9, adjust=False)
    .mean()
)

dataset["macd_hist"] = (
    dataset["macd"]
    - dataset["macd_signal"]
)

# =====================================================
# ATR
# =====================================================

dataset["atr"] = (
    (gold["High"] - gold["Low"])
    .rolling(14)
    .mean()
)

dataset["atr_pct"] = (
    dataset["atr"]
    / gold["Close"]
)

# =====================================================
# BREAKOUTS
# =====================================================

dataset["max20"] = gold["Close"].rolling(20).max()
dataset["max50"] = gold["Close"].rolling(50).max()
dataset["max250"] = gold["Close"].rolling(250).max()

dataset["breakout20"] = (
    gold["Close"] >= dataset["max20"]
).astype(int)

dataset["breakout50"] = (
    gold["Close"] >= dataset["max50"]
).astype(int)

dataset["dist_max250"] = (
    gold["Close"]
    / dataset["max250"]
    - 1
)
dataset["ath"] = gold["Close"].cummax()

dataset["dist_ath"] = (
    gold["Close"]
    / dataset["ath"]
    - 1
)

# =====================================================
# GAP
# =====================================================

dataset["gap"] = (
    gold["Open"]
    / gold["Close"].shift(1)
    - 1
)

# =====================================================
# VELAS
# =====================================================

dataset["body_pct"] = (
    abs(gold["Close"] - gold["Open"])
    / gold["Close"]
)

dataset["upper_shadow"] = (
    (
        gold["High"]
        - gold[["Open", "Close"]].max(axis=1)
    )
    / gold["Close"]
)

dataset["lower_shadow"] = (
    (
        gold[["Open", "Close"]].min(axis=1)
        - gold["Low"]
    )
    / gold["Close"]
)

rango = gold["High"] - gold["Low"]

dataset["close_position"] = np.where(
    rango == 0,
    0.5,
    (gold["Close"] - gold["Low"]) / rango
)

# =====================================================
# MOMENTUM
# =====================================================

dataset["momentum_50"] = (
    gold["Close"]
    / gold["Close"].shift(50)
    - 1
)

dataset["momentum_100"] = (
    gold["Close"]
    / gold["Close"].shift(100)
    - 1
)

dataset["momentum_200"] = (
    gold["Close"]
    / gold["Close"].shift(200)
    - 1
)

# =====================================================
# RELACIONES IMPORTANTES
# =====================================================

dataset["gold_silver_ratio"] = (
    dataset["oro_price"]
    / dataset["silver"]
)

dataset["gold_oil_ratio"] = (
    dataset["oro_price"]
    / dataset["oil"]
)

dataset["gold_vs_dxy"] = (
    dataset["oro_price"]
    / dataset["dxy"]
)

vix_ma20 = dataset["vix"].rolling(20).mean()

dataset["vix_trend"] = (
    dataset["vix"]
    / vix_ma20
    - 1
)

# =====================================================
# CRUCES DE MEDIAS
# =====================================================

dataset["golden_cross"] = (
    dataset["ma50"] > dataset["ma200"]
).astype(int)

dataset["ma20_above_ma50"] = (
    dataset["ma20"] > dataset["ma50"]
).astype(int)

dataset["ma10_above_ma20"] = (
    dataset["ma10"] > dataset["ma20"]
).astype(int)


dataset["mes"] = dataset.index.month

dataset["dia_semana"] = (
    dataset.index.dayofweek
)

dataset["trimestre"] = (
    (dataset.index.month - 1) // 3 + 1
)
dataset["corr_sp500"] = (
    dataset["oro_price"]
    .pct_change()
    .rolling(60)
    .corr(
        dataset["sp500"].pct_change()
    )
)

dataset["corr_dxy"] = (
    dataset["oro_price"]
    .pct_change()
    .rolling(60)
    .corr(
        dataset["dxy"].pct_change()
    )
)

dataset["corr_silver"] = (
    dataset["oro_price"]
    .pct_change()
    .rolling(60)
    .corr(
        dataset["silver"].pct_change()
    )
)

# =====================================================
# TARGET
# =====================================================

dataset["target_next"] = (
    gold["Close"]
    .shift(-10)
    / gold["Close"]
    - 1
)

dataset["target_buy"] = (
    dataset["target_next"] > 0.05
).astype(int)

# =====================================================
# LIMPIEZA
# =====================================================

dataset = dataset.reset_index()

dataset = dataset.rename(
    columns={
        "Date": "fecha"
    }
)

dataset = dataset.replace(
    [np.inf, -np.inf],
    np.nan
)

dataset = dataset.dropna()

# =====================================================
# ESTADISTICAS
# =====================================================

print()
print("=" * 60)
print("DATASET GENERADO")
print("=" * 60)

print()
print("Filas:", len(dataset))
print("Columnas:", len(dataset.columns))

print()
print("Target:")
print(dataset["target_buy"].value_counts())

print()
print(dataset["target_buy"].value_counts(normalize=True))

# =====================================================
# GUARDAR
# =====================================================

dataset.to_csv(
    "oro_dataset.csv",
    index=False,
    encoding="utf-8"
)

print()
print("✅ Dataset generado")
print("✅ Archivo: oro_dataset.csv")