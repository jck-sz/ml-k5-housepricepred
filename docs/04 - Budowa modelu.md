# 4. Budowa modelu

- [4. Budowa modelu](#4-budowa-modelu)
  - [4.1 Wybór algorytmu](#41-wybór-algorytmu)
  - [4.2 Podział danych](#42-podział-danych)
  - [4.3 Strojenie hiperparametrów](#43-strojenie-hiperparametrów)
    - [4.3.1 Przestrzeń przeszukiwań](#431-przestrzeń-przeszukiwań)
    - [4.3.2 Metodologia strojenia](#432-metodologia-strojenia)
    - [4.3.3 Najlepsze parametry](#433-najlepsze-parametry)
  - [4.4 Trenowanie modelu](#44-trenowanie-modelu)
  - [4.5 Analiza ważności cech](#45-analiza-ważności-cech)
  - [4.6 Zapisywanie modelu](#46-zapisywanie-modelu)

## 4.1 Wybór algorytmu

Do rozwiązania problemu regresji cen nieruchomości wybrano algorytm **lasu losowego** (ang. *Random Forest*). Jest to metoda uczenia zespołowego, która łączy predykcje wielu drzew decyzyjnych w celu uzyskania bardziej dokładnych i stabilnych wyników.

Główne zalety wybranego algorytmu:
- Odporność na przeuczenie dzięki mechanizmowi baggingu
- Brak konieczności skalowania danych
- Możliwość obsługi zarówno cech numerycznych jak i kategorycznych
- Wbudowany mechanizm oceny ważności cech
- Dobre wyniki dla problemów regresji z nieliniowymi zależnościami

## 4.2 Podział danych

Przetworzony zbiór danych został podzielony na:
- **Zbiór treningowy**: 80% danych (1138 próbek)
- **Zbiór walidacyjny**: 20% danych (284 próbki)

Podział został wykonany z zachowaniem stałego ziarna losowości (`random_state=2137`) w celu zapewnienia reprodukowalności wyników. Dodatkowo, oba zbiory zostały zapisane do plików CSV w katalogu `datasets/used/` dla możliwości późniejszej analizy:
- `datasets/used/training.csv` - zbiór treningowy
- `datasets/used/validation.csv` - zbiór walidacyjny

Moduł odpowiedzialny za trenowanie modelu to [`train_model.py`](/src/models/train_model.py).

## 4.3 Strojenie hiperparametrów

### 4.3.1 Przestrzeń przeszukiwań

W celu znalezienia optymalnych hiperparametrów modelu wykorzystano metodę przeszukiwania siatki (*GridSearchCV*). Przeszukiwana przestrzeń parametrów obejmowała:

| Parametr | Testowane wartości | Opis |
|----------|-------------------|------|
| `n_estimators` | [100, 200, 300] | Liczba drzew w lesie |
| `max_depth` | [10, 20, 30, None] | Maksymalna głębokość drzewa |
| `min_samples_split` | [2, 5, 10] | Minimalna liczba próbek do podziału węzła |
| `min_samples_leaf` | [1, 2, 4] | Minimalna liczba próbek w liściu |
| `max_features` | ['sqrt', 'log2', None] | Liczba cech do rozważenia przy podziale |
| `bootstrap` | [True, False] | Czy stosować próbkowanie ze zwracaniem |

Łączna liczba testowanych kombinacji: **648**

### 4.3.2 Metodologia strojenia

Strojenie hiperparametrów przeprowadzono z wykorzystaniem:
- **5-krotnej walidacji krzyżowej** (5-fold cross-validation)
- **Metryki optymalizacji**: ujemny błąd średniokwadratowy (negative MSE)
- **Całkowita liczba dopasowań**: 3240 (648 kombinacji × 5 foldów)

### 4.3.3 Najlepsze parametry

Po przeprowadzeniu procesu strojenia, który trwał około 15 minut na procesorze AMD Ryzen 7 7800X3D, znalezione zostały następujące optymalne parametry:

```python
{
    'bootstrap': False,
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 300
}
```

Najlepszy wynik RMSE w walidacji krzyżowej: **$22,174.39**

Warto zaznaczyć, że biblioteka scikit-learn nie wspiera akceleracji sprzętowej GPU, przez co proces trenowania odbywał się wyłącznie z wykorzystaniem procesora.

## 4.4 Trenowanie modelu

Model został wytrenowany na zbiorze treningowym z wykorzystaniem znalezionych optymalnych hiperparametrów. Proces trenowania obejmował:
1. Wczytanie przetworzonego zbioru danych z zastosowanym kodowaniem hierarchicznym
2. Podział na cechy (X) i zmienną docelową (y)
3. Dalsze zakodowanie zmiennych kategorycznych metodą *one-hot encoding* (po wcześniejszym kodowaniu hierarchicznym)
4. Trenowanie modelu Random Forest z optymalnymi parametrami
5. Ewaluację na zbiorze walidacyjnym

Warto zaznaczyć, że dane przed trenowaniem przeszły przez:
- Usunięcie wartości odstających (IQR × 2.0)
- Imputację braków (mediana dla numerycznych, dominanta dla kategorycznych)
- Kodowanie hierarchiczne dla zmiennych z naturalnym porządkiem
- Finalne kodowanie one-hot dla pozostałych zmiennych kategorycznych

Po zakończeniu trenowania model został poddany szczegółowej ewaluacji, której wyniki opisano w osobnym dokumencie.

## 4.5 Analiza ważności cech

Model Random Forest umożliwia analizę ważności poszczególnych cech w procesie predykcji. Poniższa tabela przedstawia 20 najważniejszych cech:

| Pozycja | Cecha | Ważność |
|---------|-------|---------|
| 1 | QualityPriceInteraction | 0.1004 |
| 2 | TotalSF | 0.0868 |
| 3 | OverallQual | 0.0654 |
| 4 | GrLivArea | 0.0561 |
| 5 | ExterQual | 0.0410 |
| 6 | KitchenQual | 0.0384 |
| 7 | TotalBsmtSF | 0.0349 |
| 8 | OverallQualityScore | 0.0339 |
| 9 | GarageCapacity | 0.0320 |
| 10 | GarageCars | 0.0304 |
| 11 | GarageArea | 0.0301 |
| 12 | TotalBathrooms | 0.0288 |
| 13 | YearBuilt | 0.0283 |
| 14 | 1stFlrSF | 0.0250 |
| 15 | GarageYrBlt | 0.0221 |
| 16 | HouseAge | 0.0211 |
| 17 | BsmtQual | 0.0204 |
| 18 | FullBath | 0.0179 |
| 19 | Fireplaces | 0.0172 |
| 20 | YearRemodAdd | 0.0171 |

Analiza ważności pokazuje, że najistotniejszymi cechami są:
- **QualityPriceInteraction** - cecha inżynierowana łącząca jakość z powierzchnią
- **TotalSF** - całkowita powierzchnia nieruchomości
- **OverallQual** - ogólna jakość materiałów budowlanych

Warto zauważyć, że wśród 10 najważniejszych cech znajdują się 4 cechy inżynierowane (QualityPriceInteraction, TotalSF, OverallQualityScore, GarageCapacity), co potwierdza wartość przeprowadzonego procesu inżynierii cech.

## 4.6 Zapisywanie modelu

Wytrenowany model oraz metadane zostały zapisane w następujących plikach:
- `model/house_price_model.pkl` - serializowany model w formacie joblib
- `model/model_metadata.json` - metadane zawierające parametry, metryki i datę trenowania
- `model/feature_importance.csv` - szczegółowa analiza ważności wszystkich cech
- `evaluation/validation_predictions.csv` - predykcje dla zbioru walidacyjnego
- `evaluation/validation_actual.csv` - rzeczywiste wartości dla zbioru walidacyjnego

Taka struktura zapisu umożliwia łatwe wczytanie modelu do predykcji oraz przeprowadzenie szczegółowej analizy jego działania.