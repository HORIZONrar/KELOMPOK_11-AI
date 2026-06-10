# UNTUK CHECKPOINT 4
# TRAINING MODEL RANDOM FOREST

# 1. IMPORT LIBRARY
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# 2. IMPORT DATASET

print("===== IMPORT DATASET =====")

df = pd.read_csv(
    "Sleep_health_and_lifestyle_dataset.csv",
    keep_default_na=False
)
print("Jumlah data:", df.shape)
print("\n===== CEK TARGET =====")
print(df["Sleep Disorder"].value_counts(dropna=False))

# 3. PREPROCESSING SINGKAT

print("\n===== PREPROCESSING =====")

# Hapus duplikat
df = df.drop_duplicates()

# Pisahkan Blood Pressure
df[['Systolic', 'Diastolic']] = (
    df['Blood Pressure']
    .str.split('/', expand=True)
    .astype(int)
)

df = df.drop(columns=['Blood Pressure'])

# Hapus missing value
df = df.dropna()

# Encoding
gender_encoder = LabelEncoder()
bmi_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df['Gender'] = gender_encoder.fit_transform(df['Gender'])

df['BMI Category'] = bmi_encoder.fit_transform(
    df['BMI Category']
)

df['Sleep Disorder'] = target_encoder.fit_transform(
    df['Sleep Disorder']
)

print("Preprocessing selesai")


# 4. FEATURE SELECTION

print("\n===== FEATURE SELECTION =====")

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
]]

y = df['Sleep Disorder']

print("Jumlah fitur:", X.shape[1])


# 5. FEATURE SCALING

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# 6. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData Training :", len(X_train))
print("Data Testing  :", len(X_test))


# 7. TRAINING RANDOM FOREST

print("\n===== TRAINING MODEL =====")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("Training selesai")


# 8. PREDIKSI

print("\n===== PREDIKSI =====")

y_pred = rf_model.predict(X_test)

print("Prediksi berhasil")


# 9. ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\n===== HASIL ACCURACY =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy*100:.2f}%")


# 10. CLASSIFICATION REPORT

print("\n===== CLASSIFICATION REPORT =====")

report = classification_report(y_test, y_pred)

print(report)


# 11. CONFUSION MATRIX

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

print("File berhasil disimpan:")
print("confusion_matrix.png")


# 12. SIMPAN MODEL

print("\n===== SIMPAN MODEL =====")

# Simpan model Random Forest
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

# Simpan scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Simpan encoder Gender
with open("gender_encoder.pkl", "wb") as file:
    pickle.dump(gender_encoder, file)

# Simpan encoder BMI Category
with open("bmi_encoder.pkl", "wb") as file:
    pickle.dump(bmi_encoder, file)

# Simpan encoder Target
with open("target_encoder.pkl", "wb") as file:
    pickle.dump(target_encoder, file)

print("Model berhasil disimpan")
print("Nama file:")
print("- random_forest_model.pkl")
print("- scaler.pkl")
print("- gender_encoder.pkl")
print("- bmi_encoder.pkl")
print("- target_encoder.pkl")


# 13. FEATURE IMPORTANCE

print("\n===== FEATURE IMPORTANCE =====")

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(importance)


# 14. SIMPAN GRAFIK FEATURE IMPORTANCE

plt.figure(figsize=(8,5))

plt.bar(
    importance['Feature'],
    importance['Importance']
)

plt.xticks(rotation=45)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

print("File berhasil disimpan:")
print("feature_importance.png")


# ============================================
# SELESAI
# ============================================

print("\nCheckpoint 4 selesai")
print(target_encoder.classes_)