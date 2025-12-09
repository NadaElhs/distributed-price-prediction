import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# 1. Charger le dataset
# -----------------------------
df = pd.read_csv("cleaned_data.csv")

# -----------------------------
# 2. Définir features et target
# -----------------------------
X = df.drop('price_dh', axis=1)
y = df['price_dh']

# -----------------------------
# 3. Préprocessing
# -----------------------------
num_cols = ['surface', 'bedroom', 'bathroom']
cat_cols = ['proprety_type', 'address', 'city', 'principale']

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# -----------------------------
# 4. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Modèle Linear Regression
# -----------------------------
model = Pipeline([
    ('preprocess', preprocess),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("📌 Linear Regression")
print(f"MAE  : {mae:,.0f} DH")
print(f"R²   : {r2:.3f}")

# -----------------------------
#