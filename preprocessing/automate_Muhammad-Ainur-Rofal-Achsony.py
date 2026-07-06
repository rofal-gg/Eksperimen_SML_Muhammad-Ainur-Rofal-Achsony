"""
automate_Muhammad-Ainur-Rofal-Achsony.py

Skrip otomatisasi preprocessing dataset Pima Indians Diabetes.
Dikonversi dari tahapan eksperimen manual pada notebook
`Eksperimen_Muhammad-Ainur-Rofal-Achsony.ipynb`.

Fungsi utama: preprocess_data() menerima path dataset mentah dan
mengembalikan data yang sudah siap dilatih (X_train, X_test, y_train, y_test)
sekaligus menyimpan hasilnya ke folder `diabetes_preprocessing/`.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Kolom yang tidak mungkin bernilai 0 secara medis -> 0 dianggap missing value
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    df = pd.read_csv(path)
    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Mengganti nilai 0 yang tidak masuk akal secara medis dengan NaN,
    lalu mengisinya dengan median per kolom."""
    df = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df


def remove_outliers_iqr(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Menghapus outlier ekstrem menggunakan metode IQR (1.5x) pada kolom numerik terpilih."""
    df = df.copy()
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def preprocess_data(raw_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline preprocessing lengkap:
    1. Load data
    2. Bersihkan missing value tersembunyi (0 -> median)
    3. Buang outlier ekstrem (IQR)
    4. Split train/test
    5. Scaling fitur numerik (StandardScaler)
    6. Simpan hasil ke output_dir sebagai CSV siap latih + scaler
    """
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(raw_path)
    df = clean_missing_values(df)
    # Catatan: Insulin & SkinThickness punya proporsi missing (nilai 0) yang sangat besar
    # pada dataset ini, sehingga distribusinya terpampat setelah imputasi median dan IQR
    # jadi tidak reliable untuk kolom tersebut. Outlier removal difokuskan pada kolom yang
    # distribusinya tetap representatif setelah pembersihan.
    df = remove_outliers_iqr(df, ["BloodPressure", "BMI"])

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    train_df = X_train_scaled.copy()
    train_df["Outcome"] = y_train.values
    test_df = X_test_scaled.copy()
    test_df["Outcome"] = y_test.values

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    print(f"[OK] Preprocessing selesai. Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"[OK] Hasil disimpan di: {output_dir}")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing otomatis dataset Diabetes")
    parser.add_argument("--raw_path", type=str, default="../diabetes_raw/diabetes_raw.csv",
                         help="Path ke file dataset mentah")
    parser.add_argument("--output_dir", type=str, default="diabetes_preprocessing",
                         help="Folder output hasil preprocessing")
    args = parser.parse_args()

    preprocess_data(args.raw_path, args.output_dir)
