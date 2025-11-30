import streamlit as st
import pandas as pd
import pickle

st.title("Solar Power Generation Prediction App")

with open('model.pkl','rb') as f:
    model = pickle.load(f)

cols = ['distance_to_solar_noon','temperature','wind_direction','wind_speed','sky_cover',
        'visibility','humidity','average_wind_speed-(period)','average_pressure-(period)']

inputs = []
for c in cols:
    inputs.append(st.number_input(c, 0.0))

if st.button("Predict"):
    df = pd.DataFrame([inputs], columns=cols)
    pred = model.predict(df)[0]
    st.success(f"Predicted Power Generated: {pred}")