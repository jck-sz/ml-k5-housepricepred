**Wymagania projektowe:**

---

### **Zrobione:**

#### **- Przekształcenia danych (standaryzacja nie została zrobiona, bo w RandomForest nie jest wymagana)**
- *Kodowanie zmiennych kategorycznych zrobione za pomocą one hot encoder -> pandas.get_dummies() w predict.py*
- *Imputacja braków w datasecie za pomocą SimpleImputer przy użyciu strategii "median" dla wartości numerycznych i "most_frequent" dla wartości kategorycznych*

#### **- Wybór algorytmu ML**
- *Wybrano RandomForest*

#### **- Podział datasetu na dataset treningowy**
- *Podzielony 80/20 na trening/walidacja przy użyciu scikit->train_test_split() oraz dataset testowy*

---

### **Do zrobienia:**

#### **- Cel projektu**
- *Jasno zdefiniować problem*

#### **- Opis problemu**
- *Krótko wprowadzić w temat*
- *Uzasadnić wybór problemu - dlaczego warto go rozwiązać?*
- *Wyspecyfikować źródło danych (link do Kaggle) i jego charakterystykę*

#### **- Wstępna analiza danych**
- *Rozkład, brakujące dane, korelacje*

#### **- Wizualizacja danych**
- *Wykresy, mapy cieplne, histogramy, wykresy rozrzutu itp.*

#### **- Hyperparametryzacja**
- *Wybrano jedynie hyperparametr `random_state` o wartości 2137 dla RandomForestRegressor*
- *Do poeksperymentowania i "fine tuningu" można zmienić następujące:*
  - *Warto użyć np. GridSearchCV do "fine tuningu" i znalezienia odpowiednich parametrów które określają dopasowanie modelu (np. niskie RMSE, MAE, wysokie R²)*
  - [Sklearn RandomForestRegressor dokumentacja](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    - `n_estimators` (val: 100 - 500)  
    - `max_depth` (val: None; 10-50)  
    - `max_features` (val: 'auto', 'sqrt', 'log2', None)  
    - `min_samples_split` (val: 2, 5, 10)  
    - `min_samples_leaf` (val: 1, 2, 4)  
    - `bootstrap` (val: True, False)

#### **- Ewaluacja modelu**
- *Przy fine tuningu liczymy RMSE, MAE, R²*
- *Dla problemu regresji nie nadają się: accuracy, precision, recall, F1 Score, ROC AUC*
- *Można zrobić inny model jeśli komuś się chce i porównać działanie między nimi – oznaczone jako opcjonalne*
- *Prezentacja wyników w formie wizualnej (macierze pomyłek, krzywe ROC, wykresy błędów itd.)*
  - *Na przykład:*
    - Scatter plot (oś OX: `SalePrice` z datasetu testowego, oś OY: `SalePrice` z predykcji modelu)
    - Residual plot
  - *Do wykonania w Matplotlib albo przez eksport do Excela + zabawa ręczna*

#### **- Wizualizacja wyników**
- *Interaktywne lub statyczne wizualizacje wyników i wniosków (np. z wykorzystaniem Matplotlib, Seaborn, Plotly, Tableau)*
- *Wizualizacja działania modelu (np. feature importance, SHAP, LIME, PCA/TSNE do redukcji wymiarowości)*  
  - _(nie mam pojęcia o co chodzi – do doczytania)_

#### **- Wnioski i rekomendacje**
- *Podsumowanie najważniejszych wniosków*
- *Ocena skuteczności modeli*
- *Możliwe kierunki dalszego rozwoju projektu*
