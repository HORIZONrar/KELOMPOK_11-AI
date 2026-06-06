#1.IMPORT LIBRARY
import pandas as pd
import numpy as np

print("hello world")
# LabelEncoder digunakan untuk mengubah data teks menjadi angka
# StandardScaler digunakan untuk menyamakan skala data
from sklearn.preprocessing import LabelEncoder, StandardScaler

# train_test_split digunakan untuk membagi data training dan testing
from sklearn.model_selection import train_test_split


#2.IMPORT DATASET
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# menampilkan 5 data pertama
print("===== DATA AWAL =====")
print(df.head())

# melihat jumlah baris dan kolom
print("\nJumlah baris dan kolom:", df.shape)

# melihat tipe data setiap kolom
print("\n===== TIPE DATA =====")
print(df.dtypes)


#3.MENGECEK DAN MENGHAPUS DATA DUPLIKAT

print("\n===== CEK DATA DUPLIKAT =====")
print("Jumlah data duplikat:", df.duplicated().sum())

# menghapus data duplikat
df = df.drop_duplicates()

# merapikan kembali index
df = df.reset_index(drop=True)

print("Jumlah data setelah duplikat dihapus:", df.shape)


#4.DATA CLEANING

# mengecek missing value
print("\n===== CEK MISSING VALUE =====")
print(df.isnull().sum())

# kolom Blood Pressure masih berbentuk teks
# contoh: 120/80

# memisahkan menjadi 2 kolom:
# Systolic = tekanan darah atas
# Diastolic = tekanan darah bawah

df[['Systolic', 'Diastolic']] = (
    df['Blood Pressure']
    .str.split('/', expand=True)
    .astype(int)
)

# menghapus kolom Blood Pressure lama
df = df.drop(columns=['Blood Pressure'])

# menghapus missing value jika ada
df = df.dropna()

print("\nJumlah data setelah cleaning:", df.shape)


#5.INFORMASI DATASET

print("\n===== INFORMASI DATA =====")
print(df.info())


#6.ENCODING DATA KATEGORIKAL

print("\n===== ENCODING DATA =====")

# menampilkan data sebelum encoding
print("\nSebelum encoding:")
print(df[['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']].head())

# membuat objek LabelEncoder
le = LabelEncoder()

# daftar kolom kategorikal
categorical_columns = [
    'Gender',
    'Occupation',
    'BMI Category',
    'Sleep Disorder'
]

# proses encoding
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# menampilkan hasil encoding
print("\nSesudah encoding:")
print(df[['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']].head())


#7.FEATURE SELECTION

print("\n===== FEATURE SELECTION =====")

# memilih fitur yang digunakan untuk prediksi
X = df[[
    'Gender',
    'Age',
    'Sleep Duration',
    'Physical Activity Level',
    'Stress Level',
    'BMI Category',
    'Heart Rate',
    'Daily Steps',
    'Systolic',
    'Diastolic'
]].copy()

# target yang akan diprediksi
# Sleep Disorder:
# 0 = None
# 1 = Insomnia
# 2 = Sleep Apnea

y = df['Sleep Disorder']

print("Fitur yang digunakan:")
print(X.columns.tolist())

print("\nJumlah fitur:", X.shape[1])

print("\nDistribusi target:")
print(y.value_counts())


#8.FEATURE SCALING

print("\n===== FEATURE SCALING =====")

# menampilkan data sebelum scaling
print("\nSebelum scaling:")
print(X.head())

# membuat objek scaler
scaler = StandardScaler()

# melakukan scaling
X_scaled = scaler.fit_transform(X)

# mengubah hasil scaling menjadi dataframe
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# menampilkan hasil scaling
print("\nSesudah scaling:")
print(X_scaled.head().round(4))


#9.PEMBAGIAN DATA TRAINING DAN TESTING

print("\n===== TRAIN TEST SPLIT =====")

# membagi data:
# 80% training
# 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# menampilkan jumlah data
print("Jumlah data training :", len(X_train))
print("Jumlah data testing  :", len(X_test))

# menampilkan bentuk data
print("\nShape X_train:", X_train.shape)
print("Shape X_test :", X_test.shape)


#10.PERBANDINGAN SEBELUM DAN SESUDAH

print("\n===== PERBANDINGAN DATA =====")

print(f"""
{'='*50}
{'ITEM':<30} {'SEBELUM':<10} {'SESUDAH'}
{'='*50}
{'Jumlah baris':<30} {'374':<10} {len(df)}
{'Jumlah fitur':<30} {'13':<10} {X.shape[1]}
{'Data duplikat':<30} {'Ada':<10} {'Dihapus'}
{'Missing value':<30} {'Belum dicek':<10} {'Sudah dibersihkan'}
{'Blood Pressure':<30} {'Teks':<10} {'Numerik'}
{'Data kategorikal':<30} {'Teks':<10} {'Encoded'}
{'Skala fitur':<30} {'Berbeda':<10} {'Seragam'}
{'Status data':<30} {'Mentah':<10} {'Siap dipakai'}
{'='*50}
""")

print("Preprocessing selesai!")
print("Data siap digunakan untuk model Random Forest")

