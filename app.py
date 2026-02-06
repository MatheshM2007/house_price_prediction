import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Home Price Prediction - Random Forest",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Home Price Prediction")
st.write("### Random Forest Regression")

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    return pd.read_csv("regression_home_prices.csv")

data = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# ---------------- Prepare Data ----------------
TARGET = data.columns[-1]   # last column is target

X = data.drop(TARGET, axis=1)
y = data[TARGET]

# ---------------- Train Model ----------------
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model(X, y)

# ---------------- User Input ----------------
st.subheader("🔢 Enter House Details")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(
        f"{col}",
        value=float(data[col].mean())
    )

input_df = pd.DataFrame([input_data])

# ---------------- Prediction ----------------
if st.button("💰 Predict House Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"🏷️ Predicted House Price: ₹ {prediction:,.2f}")
