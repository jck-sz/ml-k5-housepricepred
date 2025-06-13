
[🏠 ← Powrót do README](../README.md)

# 2. Źródło danych (dataset)

- [2. Źródło danych (dataset)](#2-źródło-danych-dataset)
  - [2.1 Analiza bazowego datasetu](#21-analiza-bazowego-datasetu)
    - [2.1.1 Rozkład zmiennej estymowanej](#211-rozkład-zmiennej-estymowanej)
    - [2.1.2 Brakujące wartości](#212-brakujące-wartości)
    - [2.1.3 Korelacje](#213-korelacje)
    - [2.1.4 Typy kolumn](#214-typy-kolumn)


W projekcie wykorzystano źródło danych w postaci datasetu [*House Prices - Advanced Regression Techniques*](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) dostępnego w serwisie *Kaggle*, z którego do zarówno trenowania jak i walidacji modelu wykorzystano plik [`ames-train.csv`](/datasets/ames-train.csv) (podzielony oczywiście dalej na część treningową i walidacyjną). Rozważany plik zawierał łącznie **1460 próbek** z cenami i cechami sprzedanych nieruchomości, które zebrano w mieście Ames w stanie Iowa (Stany Zjednoczone). Stąd predykcje wykonywane przez tworzony model będą reprezentatywne dla tejże lokacji oraz używać będą miar imperialnych powszechnie używanych w tamtym rejonie świata.

Dla każdej nieruchomości w opisywanym zbiorze od pyrzpisano **80 szczegółowych cech** takich jak m.in.: rok sprzedaży, kształt i typ budynku, dostępne w okolicy usługi, typ ogrzewania, powierzchnia poszczególnych pięter, liczba kominków czy łazieniek i wiele innych. 
Pozostałe, obecne domyślnie w datasecie cechy zostały opisane w pliku [`ames-data_description.txt`](/datasets/ames-data_description.txt).

## 2.1 Analiza bazowego datasetu

Bazowy dataset (przed preprocessingiem) poddano dokładniej analizie w celu uzyskania informacji niezbędnych do poprawnego przeprowadzenia preprocessingu. Użyto w przy tym stworzonego do tego celu skryptu [`analyze_datasets.py`](/src/dataset_analysis/analyze_dataset.py), który generuje szczegółowy raport na temat datasetu w formacie TXT ([`base-dataset-report.txt`](/datasets/base-dataset-report.txt)). Najważniejsze wyniki analizy zostały przedstawione w kolejnych podsekcjach.

### 2.1.1 Rozkład zmiennej estymowanej

Niezwykle istotnym elementem analizy datasetu było wygenerowanie histogramu pokazującego rozkład wartości estymowanej zmiennej (ang. *target*), będącej wartością numeryczną. Zamieszczono go na rysunku poniżej wraz z tabelą zawierającą miary statystyczne dotyczące rozkładu.

![histogram rozkładu zmiennej estymowanej](/datasets/plots/base/target_histogram.png)

| Miara statystyczna     | Wartość [$]  |
| ---------------------- | ------------:|
| Minumum                |    34,900.00 |
| Średnia                |   180,921.20 |
| Maksimum               |   755,000.00 |
| Odchylenie standardowe |    79,442.50 |

Należy zauważyć, że rozkład jest przechylony prawostronnie, a znacząca większość próbek tyczy się nieruchomości o wartościach między 100 a 300 tysięcy USD. Skąpo reprezentowane są zaś nieruchomości o wartości większej niż 400 tysięcy USD (zaledwie około 30 próbek).

### 2.1.2 Brakujące wartości

W ramach analizy zbadano także isntienie brakujących wartości w kolumnach datasetu bazowego. Okazało się, że stanowią one 6.62 % wszystkich wartości w zbiorze danych. Procentowy udział pustych wartości w każdej z kolumn przedstawiono na rysunku poniżej. Po głębszym przeanalizowaniu interpretacji kolumn okazuje się, że większość brakujących wartości znajduje się w kolumnach o typach kategorycznych, takich jak `MiscFeature` czy `Alley` czy `Fence`. Oznaczają one przy tym brak jakiegoś rodzaju udogodnienia, przykładowo: asfaltowej drogi dojazdowej czy płotu. 

![wykaz brakujących wartości](/datasets/plots/base/null-values.png)

### 2.1.3 Korelacje

Dokonano także analizy nieliniowych korelacji (Spearmana) pomiędzy liczbowymi cechami datasetu oraz pomiędzy liczbowymi cechami a zmienną docelową. W wyniku tego wygenerowano mapę ciepła przedstawioną poniżej. Należy zauważyć, że pomiędzy niektórymi uwzględnionymi na niej cechami (np. `GarageYrBlt`-`YearBuilt`) występują bardzo silne (> 0.80), nielinowe korelacje.

![mapa ciepła korelacji nieliniowych](/datasets/plots/base/correlations.png)

### 2.1.4 Typy kolumn

Na poniższym rysunku zamieszczono udział poszczególnych typów kolumn w bazowym zbiorze danych. Należy zauważyć, że największą grupę tworzą w nim kolumny kategoryczne, przedstawiane w formie obiektów przechowujących ciągi znaków. Pozostałe kolumny zawierają wartości liczbowe.

![types pie chart](/datasets/plots/base/types.png)

[🏠 ← Powrót do README](../README.md)
