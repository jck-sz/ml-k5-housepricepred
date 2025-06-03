# 🏠 House Price Prediction (Ames Housing Data)
# ml-k5-housepricepred repo
This project predicts house prices using the Ames, Iowa housing dataset. It includes data preprocessing, feature engineering, model training, and a web interface for user input and price prediction.

---

## 📁 Project Structure

```
.
├── app/                      # Streamlit or Flask frontend app for predictions
│   └── app.py
│
├── datasets/                # Raw and processed data files
│   ├── processed/           # Output of cleaned data
│   │   └── ames-train-clean.csv
│   ├── ames-train.csv       # Raw training dataset
│   ├── ames-test.csv        # Raw test dataset
│   └── ames-data_description.txt  # Column descriptions from dataset source
│
├── logs/                    # Log files generated during preprocessing
│   └── preprocess.log
│
├── model/                   # Trained model artifacts (e.g., .pkl files)
│   └── house_price_model.pkl (to be generated and ignored by Git)
│
├── src/                     # Source code for the ML pipeline
│   ├── data_preprocessing/  # Data cleaning scripts
│   │   └── preprocess.py
│   ├── features/            # Feature engineering code
│   │   └── build_features.py
│   ├── models/              # Model training and prediction
│   │   ├── train_model.py
│   │   └── predict.py
│   └── utils/               # Utility functions (e.g., logger)
│       └── logger.py
│
├── tests/                   # Unit tests for key components
│   └── test_*.py
│
├── .gitignore               # Git ignored files (e.g., .venv, logs)
├── requirements.txt         # Python dependencies
└── README.md                # Project overview (this file)
```

---

## 🚀 How to Run the Project

1. **Set up your virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # or source .venv/bin/activate on macOS/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the data**
   ```bash
   python src/data_preprocessing/preprocess.py
   ```

4. **Train the model**
   ```bash
   python src/models/train_model.py
   ```

5. **Launch the app**
   ```bash
   streamlit run app/app.py
   ```

   The Streamlit app provides a simple form to enter house details and view the predicted price. Make sure the trained model (`house_price_model.pkl`) and the cleaned dataset (`ames-train-clean.csv`) are available in the expected folders before running the app.

6. **--- ALTERNATIVELY : run all in one go---**
   ```bash
   python run_all.py
   ```
---

## 👥 Contributors

- Jacek Szlączka
- Marcin Michalak
- 
- 
- 

---

## 📌 Notes

- All model artifacts and logs are excluded from Git using `.gitignore`.
- Cleaned data is stored in `datasets/processed/`.
- Logging output (e.g., missing data imputs) goes to `logs/preprocess.log`.

---

## 🚧 TODOs

- Add unit tests under `tests/` *Optional*
- Optionally switch to `LinearRegression` model for simplicity or a different more advanced model *Optional*
- Prepare final submission script to predict on `ames-test.csv` *Optional*
