import streamlit as st
import pandas as pd
import pickle

# LOAD MODEL DAN FILE

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

# TAMPILAN WEBSITE

st.title("Sleep Disorder Prediction System")

st.write(
    """
    Aplikasi ini digunakan untuk memprediksi
    kemungkinan gangguan tidur berdasarkan
    data kesehatan dan gaya hidup pengguna.
    """
)

st.write("Model : Random Forest")

st.info(
    """
    Silakan isi data sesuai kondisi Anda.
    Hasil prediksi hanya digunakan untuk
    keperluan pembelajaran dan penelitian.
    """
)

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Home",
        "About",
        "Developer"
    ]
)

if menu == "About":

    st.title("About Application")

    st.write("""
    Sleep Disorder Prediction System merupakan aplikasi
    yang digunakan untuk memprediksi gangguan tidur
    berdasarkan data kesehatan dan gaya hidup pengguna.
    """)

    st.subheader("Dataset")

    st.write(
        "Sleep Health and Lifestyle Dataset"
    )

    st.subheader("Machine Learning Model")

    st.write(
        "Random Forest Classifier"
    )

elif menu == "Developer":

    st.title("Developer Profile")

    st.write("Nama : Kelompok 11")

    st.write("Nama : Stevano Imanuel Panjaitan (412024015)")

    st.write("Nama : Leonard Francois (412024035)")

    st.write("Program Studi : Rekayasa Perangkat Lunak")

    st.write("Mata Kuliah : Artificial Intelligence")

    st.write(
        "Aplikasi ini dibuat untuk memenuhi "
        "tugas UAS Artificial Intelligence."
    )

if menu == "Home":

 st.subheader("Input Data")

 gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
 )

 age = st.number_input(
    "Age (1 - 100 Tahun)",
    min_value=1,
    max_value=100,
    value=25
 )

 sleep_duration = st.number_input(
    "Sleep Duration (Jam)",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    help="Rata-rata lama tidur per hari"
 )

 physical_activity = st.number_input(
    "Physical Activity Level (0 - 100)",
    min_value=0,
    max_value=100,
    value=50,
    help="Tingkat aktivitas fisik harian"
 )

 stress_level = st.number_input(
    "Stress Level (1 - 10)",
    min_value=1,
    max_value=10,
    value=5,
    help="1 = sangat rendah, 10 = sangat tinggi"
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
    value=70,
    help="Detak jantung per menit"
 )

 daily_steps = st.number_input(
    "Daily Steps",
    min_value=0,
    value=5000,
    help="Jumlah langkah per hari"
 )

 systolic = st.number_input(
    "Systolic",
    min_value=50,
    max_value=250,
    value=120,
    help="Tekanan darah sistolik"
 )

 diastolic = st.number_input(
    "Diastolic",
    min_value=30,
    max_value=150,
    value=80,
    help="Tekanan darah diastolik"
 )

 # PREDIKSI

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

    if result == "None":

        st.success(
            "Tidak terdeteksi gangguan tidur."
        )

        st.write(
            """
            Kondisi tidur pengguna berada
            pada kategori normal berdasarkan
            data yang dimasukkan.
            """
        )

    elif result == "Insomnia":

        st.warning(
            "Terindikasi mengalami Insomnia."
        )

        st.write(
            """
            Insomnia merupakan gangguan tidur
            yang menyebabkan kesulitan tidur
            atau mempertahankan kualitas tidur.
            """
        )

    elif result == "Sleep Apnea":

        st.error(
            "Terindikasi mengalami Sleep Apnea."
        )

        st.write(
            """
            Sleep Apnea adalah gangguan tidur
            yang menyebabkan pernapasan berhenti
            sementara saat tidur.
            """
        )

