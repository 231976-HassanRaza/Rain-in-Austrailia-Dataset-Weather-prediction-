"""
app.py
------
Streamlit web app for predicting whether it will rain tomorrow,
using the model trained by train_model.py.

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "model"

st.set_page_config(page_title="Will it Rain Tomorrow?", page_icon="🌦️", layout="centered")

# ---------------------------------------------------------------
# Load model + preprocessors (cached so it only loads once)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    required = ["rf_model.pkl", "imputer.pkl", "scaler.pkl", "encoder.pkl", "columns.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        return None

    model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
    cols = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
    return model, imputer, scaler, encoder, cols


artifacts = load_artifacts()

st.title("🌦️ Will it Rain Tomorrow?")
st.caption("Random Forest classifier trained on the Rain in Australia dataset")

if artifacts is None:
    st.error(
        "Model files not found in the `model/` folder.\n\n"
        "Run `python train_model.py` first to train and save the model, "
        "then restart this app."
    )
    st.stop()

model, imputer, scaler, encoder, cols = artifacts
numeric_cols = cols["numeric_cols"]
categorical_cols = cols["categorical_cols"]
encoded_cols = cols["encoded_cols"]

LOCATIONS = [
    "Adelaide", "Albany", "Albury", "AliceSprings", "BadgerysCreek", "Ballarat",
    "Bendigo", "Brisbane", "Cairns", "Canberra", "Cobar", "CoffsHarbour", "Dartmoor",
    "Darwin", "GoldCoast", "Hobart", "Katherine", "Launceston", "Melbourne",
    "MelbourneAirport", "Mildura", "Moree", "MountGambier", "MountGinini", "Newcastle",
    "Nhil", "NorahHead", "NorfolkIsland", "Nuriootpa", "PearceRAAF", "Penrith", "Perth",
    "PerthAirport", "Portland", "Richmond", "Sale", "SalmonGums", "Sydney",
    "SydneyAirport", "Townsville", "Tuggeranong", "Uluru", "WaggaWagga", "Walpole",
    "Watsonia", "Williamtown", "Witchcliffe", "Wollongong", "Woomera",
]
WIND_DIRS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

st.subheader("Enter today's weather details")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", LOCATIONS, index=LOCATIONS.index("Sydney"))
    min_temp = st.number_input("Min Temp (°C)", value=15.0)
    max_temp = st.number_input("Max Temp (°C)", value=25.0)
    rainfall = st.number_input("Rainfall (mm)", value=0.0, min_value=0.0)
    evaporation = st.number_input("Evaporation (mm)", value=5.0)
    sunshine = st.number_input("Sunshine (hours)", value=7.0)
    wind_gust_dir = st.selectbox("Wind Gust Direction", WIND_DIRS, index=WIND_DIRS.index("W"))
    wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", value=35.0)
    wind_dir_9am = st.selectbox("Wind Direction 9am", WIND_DIRS, index=WIND_DIRS.index("N"))
    wind_dir_3pm = st.selectbox("Wind Direction 3pm", WIND_DIRS, index=WIND_DIRS.index("W"))
    wind_speed_9am = st.number_input("Wind Speed 9am (km/h)", value=15.0)

with col2:
    wind_speed_3pm = st.number_input("Wind Speed 3pm (km/h)", value=20.0)
    humidity_9am = st.number_input("Humidity 9am (%)", value=70.0, min_value=0.0, max_value=100.0)
    humidity_3pm = st.number_input("Humidity 3pm (%)", value=50.0, min_value=0.0, max_value=100.0)
    pressure_9am = st.number_input("Pressure 9am (hPa)", value=1015.0)
    pressure_3pm = st.number_input("Pressure 3pm (hPa)", value=1012.0)
    cloud_9am = st.number_input("Cloud 9am (oktas)", value=4.0, min_value=0.0, max_value=9.0)
    cloud_3pm = st.number_input("Cloud 3pm (oktas)", value=4.0, min_value=0.0, max_value=9.0)
    temp_9am = st.number_input("Temp 9am (°C)", value=18.0)
    temp_3pm = st.number_input("Temp 3pm (°C)", value=23.0)
    rain_today = st.selectbox("Rain Today?", ["No", "Yes"])

predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    single_input = {
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today,
    }

    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]

    pred = model.predict(X_input)[0]
    pos_idx = list(model.classes_).index("Yes")
    rain_prob = model.predict_proba(X_input)[0][pos_idx]

    st.divider()
    if pred == "Yes":
        st.success(f"🌧️ **Yes, it will likely rain tomorrow.**")
    else:
        st.info(f"☀️ **No, it likely won't rain tomorrow.**")

    st.metric("Probability of rain", f"{rain_prob * 100:.1f}%")
    st.progress(min(max(rain_prob, 0.0), 1.0))
