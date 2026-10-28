import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.simplefilter("ignore")

# ==========================================
# DATASET
# ==========================================

df = pd.read_csv(
    "oro_dataset.csv"
)

TARGET = "target_buy"

X = df.drop(
    columns=[
        "fecha",
        "target_buy",
        "target_next"
    ]
)

y = df[TARGET]

# ==========================================
# SOLO NUMERICAS
# ==========================================

X = X.select_dtypes(
    include="number"
)

# ==========================================
# IMPUTACION
# ==========================================

imputer = SimpleImputer(
    strategy="median"
)

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# ==========================================
# MODELO FINAL
# ==========================================

modelo = GradientBoostingClassifier(

    random_state=42,

    n_estimators=200,

    learning_rate=0.03,

    max_depth=3
)

print()
print("Entrenando modelo...")

modelo.fit(
    X,
    y
)

# ==========================================
# GUARDAR
# ==========================================

joblib.dump(
    modelo,
    "mejor_modelo.pkl"
)

joblib.dump(
    imputer,
    "imputer.pkl"
)

joblib.dump(
    list(X.columns),
    "features.pkl"
)

print()
print("✅ mejor_modelo.pkl")
print("✅ imputer.pkl")
print("✅ features.pkl")

# ==========================================
# IMPORTANCIA VARIABLES
# ==========================================

importancia = pd.DataFrame({

    "Variable": X.columns,

    "Importancia":
    modelo.feature_importances_
})

importancia = importancia.sort_values(
    "Importancia",
    ascending=False
)

importancia.to_csv(
    "importancia_variables.csv",
    index=False
)

print()
print("TOP 20 VARIABLES")
print()

print(
    importancia.head(20)
)

print()

print(
    "✅ importancia_variables.csv"
)

print()
print("MODELO ENTRENADO CORRECTAMENTE")