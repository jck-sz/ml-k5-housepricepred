# 🏠 Estymacja cen nieruchomości (Ames Housing Data)
# ml-k5-housepricepred repo

Ten projekt estymuje ceny nieruchomości używając datasetu z Ames, Iowa. Projekt zawiera preprocessing, inżynierię cech, trenowanie modelu oraz interfejs webowy pozwalający na wprowadzanie danych i uzyskiwanie predykcji.

---

## 📁 Struktura projektu

```
.
├── app/                            # Aplikacja korzystająca z frameworka Streamlit
│   └── app.py
│
├── datasets/                       # Dataset zawierający bazowe oraz przetworzone dane
│   ├── processed/                  # Przetworzone datasety
│   │   └── ames-train-clean.csv
│   │   └── ames-train-featured.csv
│   |   └── ames-test-featured.csv
│   ├── ames-train.csv              # Bazowy dataset do trenowania modelu
│   └── ames-data_description.txt   # Opis cech datasetu bazowego
│   └── base-dataset-report.txt     # Raport z analizy datasetu bazowego
│
├── docs/                           # Dokumentacja
│   ├── 01 - wstęp.md               # Wstęp do projektu
│   ├── 02 - dataset.md             # Opis datasetu bazowego
│   ├── 03 - preprocessing.md       # Opis procesu preprocessingu 
│   ├── 04 - Budowa modelu.md       # Opis budowy modelu
│   ├── 05 - Ewaluacja modelu.md    # Opis procesu ewaluacji modelu oraz jej wyników
|   └── 06 - Wnioski.md             # Wnioski z projektu
|
├── logs/                           # Logi z preprocessingu
│   └── preprocess.log
│
├── model/                          # Pliki wynikowe po trenowaniu modelu
│   └── house_price_model.pkl       # Wytrenowany model
│   └── model_metadata.json         # Metadane modelu
│   └── feature_importance.csv      # Ważność cech
│
├── src/                            # Kod projektu
│   ├── data_preprocessing/  
│   │   └── preprocess.py           # Skrypt preprocessingu danych
│   ├── dataset_analysis/  
│   │   └── analyze_dataset.py      # Skrypt do analizy datasetu bazowego
│   ├── features/            
│   │   └── build_features.py       # Skrypt inżynierii cech
│   ├── models/                     
│   │   └── train_model.py          # Skrypt trenowania modelu
│   └── utils/                      
│       └── logger.py               # Moduł logowania (używany w preprocessingu)
│
│
├── .gitignore                      # gitignore
├── requirements.txt                # Lista wymaganych bibliotek
├── run_all.py                      # Skrypt do uruchomienia całego projektu
├── evaluate_model.py               # Skrypt do ewaluacji modelu i generowania części wykresów
├── wymagania.md                    # Wymagania projektowe, tracking TODOs na potrzeby projektu
│
└── README.md                       # Opis projektu (ten plik)
```

---

## 🚀 Jak uruchomić projekt

1. **Aktywuj swoje środowisko wirtualne (venv)**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # lub `source .venv/bin/activate na macOS/Linux`
   ```

2. **Zainstaluj wymagane biblioteki**
   ```bash
   pip install -r requirements.txt
   ```

3. **Uruchom skrypt do preprocessingu danych**
   ```bash
   python src/data_preprocessing/preprocess.py
   ```

4. **Uruchom skrypt do trenowania modelu. Uwaga! Trenowanie modelu może potrwać do 1 godziny, w zależności od wydajności komputera.**
   ```bash
   python src/models/train_model.py
   ```

5. **Uruchom aplikację webową**
   ```bash
   streamlit run app/app.py
   ```

   Aplikacja Streamlit umożliwia wprowadzanie danych o domu i wyświetlanie przewidywanej ceny. Upewnij się, że wytrenowany model (`house_price_model.pkl`) i przetworzone datasety (`ames-train-clean.csv`, `ames-train-featured.csv`, `ames-test-featured.csv`) znajdują się w odpowiednich folderach przed uruchomieniem aplikacji.

6. **--- Alternatywnie : uruchomienie całego projektu za pomocą jednego skryptu ---**
   ```bash
   python run_all.py
   ```
---

## 👥 Zespół

- Mateusz Mierzwa
- Marcin Michalak
- Jacek Szlączka
- Dawid Waligórski
- 

---

## 📌 Uwagi

- Wszystkie artefakty modelu i logi są wykluczane z Gita za pomocą `.gitignore`.
---
