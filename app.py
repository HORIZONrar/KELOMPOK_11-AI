import streamlit as st
import pandas as pd
import pickle

# ==========================
# LOAD MODEL DAN FILE
# ==========================

with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("gender_encoder.pkl", "rb") as file:
    gender_encoder = pickle.load(file)

with open("bmi_encoder.pkl", "rb") as file:
    bmi_encoder = pickle.load(file)

with open("target_encoder.pkl", "rb") as file:
    target_encoder = pickle.load(file)

# ==========================
# TAMPILAN WEBSITE
# ==========================

st.title("Sleep Disorder Prediction")

st.write(
    "Masukkan data kesehatan dan gaya hidup untuk "
    "melakukan prediksi gangguan tidur."
)

st.subheader("Input Data")

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

sleep_duration = st.number_input(
    "Sleep Duration",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

physical_activity = st.number_input(
    "Physical Activity Level",
    min_value=0,
    max_value=100,
    value=50
)

stress_level = st.number_input(
    "Stress Level",
    min_value=1,
    max_value=10,
    value=5
)

bmi_category = st.selectbox(
    "BMI Category",
    [
        "Normal",
        "Normal Weight",
        "Obese",
        "Overweight"
    ]
)

heart_rate = st.number_input(
    "Heart Rate",
    min_value=30,
    max_value=200,
    value=70
)

daily_steps = st.number_input(
    "Daily Steps",
    min_value=0,
    value=5000
)

systolic = st.number_input(
    "Systolic",
    min_value=50,
    max_value=250,
    value=120
)

diastolic = st.number_input(
    "Diastolic",
    min_value=30,
    max_value=150,
    value=80
)

# ==========================
# PREDIKSI
# ==========================

if st.button("Predict"):

    gender_encoded = gender_encoder.transform(
        [gender]
    )[0]

    bmi_encoded = bmi_encoder.transform(
        [bmi_category]
    )[0]

    input_data = pd.DataFrame({
        "Gender": [gender_encoded],
        "Age": [age],
        "Sleep Duration": [sleep_duration],
        "Physical Activity Level": [physical_activity],
        "Stress Level": [stress_level],
        "BMI Category": [bmi_encoded],
        "Heart Rate": [heart_rate],
        "Daily Steps": [daily_steps],
        "Systolic": [systolic],
        "Diastolic": [diastolic]
    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    result = target_encoder.inverse_transform(
        prediction
    )[0]

    st.subheader("Prediction Result")

    st.success(
        f"Hasil Prediksi: {result}"
    )

