import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

# ==========================================
# CARGAR OBJETOS
# ==========================================

modelo = joblib.load(
    "mejor_modelo.pkl"
)

imputer = joblib.load(
    "imputer.pkl"
)

features = joblib.load(
    "features.pkl"
)

# ==========================================
# DESCARGA ORO
# ==========================================

print("Descargando datos...")

df = yf.download(
    "GC=F",
    start="2010-01-01",
    auto_adjust=True,
    progress=False
)

if isinstance(
    df.columns,
    pd.MultiIndex
):
    df.columns = (
        df.columns
        .get_level_values(0)
    )

# ==========================================
# FEATURES
# ==========================================

dataset = pd.DataFrame(
    index=df.index
)

dataset["oro_price"] = df["Close"]

dataset["ret_1d"] = (
    df["Close"].pct_change(1)
)

dataset["ret_3d"] = (
    df["Close"].pct_change(3)
)

dataset["ret_5d"] = (
    df["Close"].pct_change(5)
)

dataset["ret_10d"] = (
    df["Close"].pct_change(10)
)

dataset["ret_20d"] = (
    df["Close"].pct_change(20)
)

dataset["ret_50d"] = (
    df["Close"].pct_change(50)
)

# ==========================================
# MEDIAS
# ==========================================

dataset["ma10"] = (
    df["Close"]
    .rolling(10)
    .mean()
)

dataset["ma20"] = (
    df["Close"]
    .rolling(20)
    .mean()
)

dataset["ma50"] = (
    df["Close"]
    .rolling(50)
    .mean()
)

dataset["ma100"] = (
    df["Close"]
    .rolling(100)
    .mean()
)

dataset["ma200"] = (
    df["Close"]
    .rolling(200)
    .mean()
)

# ==========================================
# DISTANCIAS
# ==========================================

dataset["dist_ma10"] = (
    df["Close"]
    / dataset["ma10"]
    - 1
)

dataset["dist_ma20"] = (
    df["Close"]
    / dataset["ma20"]
    - 1
)

dataset["dist_ma50"] = (
    df["Close"]
    / dataset["ma50"]
    - 1
)

dataset["dist_ma100"] = (
    df["Close"]
    / dataset["ma100"]
    - 1
)

dataset["dist_ma200"] = (
    df["Close"]
    / dataset["ma200"]
    - 1
)

# ==========================================
# VOLATILIDAD
# ==========================================

dataset["vol_5d"] = (
    dataset["ret_1d"]
    .rolling(5)
    .std()
)

dataset["vol_20d"] = (
    dataset["ret_1d"]
    .rolling(20)
    .std()
)

dataset["vol_50d"] = (
    dataset["ret_1d"]
    .rolling(50)
    .std()
)

# ==========================================
# RSI
# ==========================================

delta = df["Close"].diff()

gain = delta.clip(
    lower=0
)

loss = -delta.clip(
    upper=0
)

avg_gain = (
    gain
    .rolling(14)
    .mean()
)

avg_loss = (
    loss
    .rolling(14)
    .mean()
)

rs = avg_gain / avg_loss

dataset["rsi"] = (
    100 -
    (
        100 /
        (1 + rs)
    )
)

# ==========================================
# MACD
# ==========================================

ema12 = (
    df["Close"]
    .ewm(
        span=12,
        adjust=False
    )
    .mean()
)

ema26 = (
    df["Close"]
    .ewm(
        span=26,
        adjust=False
    )
    .mean()
)

dataset["macd"] = (
    ema12 - ema26
)

dataset["macd_signal"] = (
    dataset["macd"]
    .ewm(
        span=9,
        adjust=False
    )
    .mean()
)

dataset["macd_hist"] = (
    dataset["macd"]
    - dataset["macd_signal"]
)

# ==========================================
# ATR
# ==========================================

dataset["atr"] = (
    (
        df["High"]
        - df["Low"]
    )
    .rolling(14)
    .mean()
)

dataset["atr_pct"] = (
    dataset["atr"]
    / df["Close"]
)

# ==========================================
# BREAKOUTS
# ==========================================

dataset["max20"] = (
    df["Close"]
    .rolling(20)
    .max()
)

dataset["max50"] = (
    df["Close"]
    .rolling(50)
    .max()
)

dataset["max250"] = (
    df["Close"]
    .rolling(250)
    .max()
)

dataset["breakout20"] = (
    df["Close"]
    >= dataset["max20"]
).astype(int)

dataset["breakout50"] = (
    df["Close"]
    >= dataset["max50"]
).astype(int)

dataset["dist_max250"] = (
    df["Close"]
    / dataset["max250"]
    - 1
)

# ==========================================
# GAP
# ==========================================

dataset["gap"] = (
    df["Open"]
    / df["Close"].shift(1)
    - 1
)

# ==========================================
# VELAS
# ==========================================

dataset["body_pct"] = (
    abs(
        df["Close"]
        - df["Open"]
    )
    / df["Close"]
)

dataset["upper_shadow"] = (
    (
        df["High"]
        - df[["Open", "Close"]]
        .max(axis=1)
    )
    / df["Close"]
)

dataset["lower_shadow"] = (
    (
        df[["Open", "Close"]]
        .min(axis=1)
        - df["Low"]
    )
    / df["Close"]
)

rango = (
    df["High"]
    - df["Low"]
)

dataset["close_position"] = np.where(
    rango == 0,
    0.5,
    (
        df["Close"]
        - df["Low"]
    ) / rango
)

# ==========================================
# MOMENTUM
# ==========================================

dataset["momentum_50"] = (
    df["Close"]
    / df["Close"].shift(50)
    - 1
)

dataset["momentum_100"] = (
    df["Close"]
    / df["Close"].shift(100)
    - 1
)

dataset["momentum_200"] = (
    df["Close"]
    / df["Close"].shift(200)
    - 1
)

# ==========================================
# CRUCES
# ==========================================

dataset["golden_cross"] = (
    dataset["ma50"]
    > dataset["ma200"]
).astype(int)

dataset["ma20_above_ma50"] = (
    dataset["ma20"]
    > dataset["ma50"]
).astype(int)

dataset["ma10_above_ma20"] = (
    dataset["ma10"]
    > dataset["ma20"]
).astype(int)

# ==========================================
# LIMPIEZA
# ==========================================

dataset = dataset.replace(
    [np.inf, -np.inf],
    np.nan
)

dataset = dataset.dropna()

# ==========================================
# ULTIMO REGISTRO
# ==========================================

ultimo = dataset.tail(1)

# ==========================================
# REINDEXADO
# ==========================================

ultimo = ultimo.reindex(
    columns=features,
    fill_value=np.nan
)

ultimo = pd.DataFrame(
    imputer.transform(
        ultimo
    ),
    columns=features
)

# ==========================================
# PREDICCION
# ==========================================

prob = modelo.predict_proba(
    ultimo
)[0, 1]

pred = modelo.predict(
    ultimo
)[0]

# ==========================================
# DATOS PARA HTML
# ==========================================

fecha = str(
    dataset.index[-1].date()
)

precio = round(
    float(
        dataset["oro_price"].iloc[-1]
    ),
    2
)

if prob >= 0.50:

    señal = "🟢 SEÑAL MUY FUERTE 🟢"
    color = "#22c55e"

elif prob >= 0.30:

    señal = "🟡 POSIBLE COMPRA 🟡"
    color = "#eab308"

elif prob >= 0.15:

    señal = "🟠 VIGILAR 🟠"
    color = "#f97316"

else:

    señal = "🔴 NO COMPRAR 🔴"
    color = "#ef4444"

# ==========================================
# HTML
# ==========================================

html = f"""
<!DOCTYPE html>
<html lang="es">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Predicción Oro IA</title>

<style>

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{

    font-family: Segoe UI, Arial, sans-serif;

    background:
    linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #111827
    );

    display: flex;

    justify-content: center;

    align-items: center;

    min-height: 100vh;

    color: white;
}}

.card {{

    width: 900px;

    background: rgba(
        31,
        41,
        55,
        0.95
    );

    border-radius: 25px;

    padding: 40px;

    box-shadow:
    0 0 50px rgba(
        0,
        0,
        0,
        0.5
    );

    text-align: center;
}}

.header {{

    margin-bottom: 30px;
}}

.header h1 {{

    color: gold;

    font-size: 42px;
}}

.subtitle {{

    color: #cbd5e1;

    margin-top: 10px;
}}

.grid {{

    display: flex;

    justify-content: center;

    gap: 30px;

    margin-top: 30px;

    flex-wrap: wrap;
}}

.box {{

    background: #111827;

    border-radius: 20px;

    padding: 25px;

    min-width: 220px;
}}

.label {{

    color: #94a3b8;

    font-size: 14px;

    text-transform: uppercase;
}}

.value {{

    margin-top: 10px;

    font-size: 32px;

    font-weight: bold;
}}

.circle {{

    width: 250px;

    height: 250px;

    border-radius: 50%;

    background:
        conic-gradient(
            {color} {prob*100}%,
            #374151 0
        );

    display: flex;

    justify-content: center;

    align-items: center;

    margin: 30px auto;
}}

.circle-inner {{

    width: 190px;

    height: 190px;

    border-radius: 50%;

    background: #111827;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 42px;

    font-weight: bold;
}}

.signal {{

    margin-top: 20px;

    font-size: 34px;

    font-weight: bold;

    color: {color};
}}

.progress-container {{

    background: #374151;

    height: 35px;

    border-radius: 20px;

    overflow: hidden;

    margin-top: 30px;
}}

.progress-bar {{

    height: 35px;

    width: {prob*100:.2f}%;

    background: {color};

    transition: width 1s ease;
}}

.footer {{

    margin-top: 30px;

    color: #94a3b8;
}}

</style>

</head>

<body>

<div class="card">

<div class="header">

<h1>📈 Predicción Oro IA</h1>

</div>

<div class="grid">

<div class="box">

<div class="label">
Fecha
</div>

<div class="value">
{fecha}
</div>

</div>

<div class="box">

<div class="label">
Precio Oro
</div>

<div class="value">
${precio}
</div>

</div>

</div>

<div class="circle">

<div class="circle-inner">

{prob:.2%}

</div>

</div>

<div class="signal">

{señal}

</div>

<div class="progress-container">

<div class="progress-bar"></div>

</div>

</div>

</body>

</html>
"""

# ==========================================
# GUARDAR HTML
# ==========================================

from pathlib import Path

# Carpeta docs
docs_dir = Path(__file__).resolve().parent / "docs"

# Crear docs si no existe
docs_dir.mkdir(
    parents=True,
    exist_ok=True
)

# Fichero HTML
html_file = docs_dir / "index.html"

with open(
    html_file,
    "w",
    encoding="utf-8"
) as f:
    f.write(html)

print()
print("✅ HTML generado")
print(f"📄 {html_file}")

# ==========================================
# RESULTADO
# ==========================================

print()
print("=" * 60)
print("PREDICCION ORO")
print("=" * 60)

print()

print(
    f"Probabilidad: {prob:.2%}"
)

print()

if prob >= 0.50:
    print("🟢 SEÑAL MUY FUERTE 🟢")

elif prob >= 0.30:
    print("🟡 POSIBLE COMPRA 🟡")

elif prob >= 0.15:
    print("🟠 VIGILAR 🟠")

else:
    print("🔴 NO COMPRAR 🔴")

print()

print(
    "Predicción:",
    pred
)
