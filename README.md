# Eksperimen_SML_Muhammad-Ainur-Rofal-Achsony

Repository ini merupakan bagian dari submission "Membangun Sistem Machine
Learning" (Dicoding) — Kriteria 1: Eksperimen terhadap Dataset Pelatihan.

**Nama:** Muhammad Ainur Rofal Achsony
**Dataset:** Pima Indians Diabetes Dataset (prediksi risiko diabetes)

## Struktur
```
Eksperimen_SML_Muhammad-Ainur-Rofal-Achsony/
├── .github/workflows/preprocessing.yml   # CI otomatisasi preprocessing (Advance)
├── diabetes_raw/diabetes_raw.csv         # Dataset mentah
└── preprocessing/
    ├── Eksperimen_Muhammad-Ainur-Rofal-Achsony.ipynb   # Notebook eksperimen manual
    ├── automate_Muhammad-Ainur-Rofal-Achsony.py        # Skrip preprocessing otomatis
    └── diabetes_preprocessing/                          # Output data siap latih
```

## Cara pakai
```bash
pip install pandas numpy scikit-learn joblib
cd preprocessing
python automate_Muhammad-Ainur-Rofal-Achsony.py
```

## Cara push ke GitHub
```bash
git init
git add .
git commit -m "Kriteria 1: eksperimen dan preprocessing dataset diabetes"
git branch -M main
git remote add origin https://github.com/rofal-gg/Eksperimen_SML_Muhammad-Ainur-Rofal-Achsony.git
git push -u origin main
```
Pastikan repository di-set **Public** di GitHub (repo Private akan membuat
submission ditolak).
