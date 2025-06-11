
[🏠 ← Powrót do README](../README.md)

# 3. Preprocessing

- [3. Preprocessing](#3-preprocessing)
  - [3.1 Wykonane operacje](#31-wykonane-operacje)
    - [3.1.1 Usunięcie wartości odstających](#311-usunięcie-wartości-odstających)
    - [3.1.2 Uzupełnienie wartości brakujących](#312-uzupełnienie-wartości-brakujących)
    - [3.1.3 Stworzenie dodatkowych cech](#313-stworzenie-dodatkowych-cech)
    - [3.1.4 Zakodowanie zmiennych kategorycznych](#314-zakodowanie-zmiennych-kategorycznych)
  - [3.2 Analiza wariantów preprocessingu](#32-analiza-wariantów-preprocessingu)


Na podstawie wykonanej uprzednio analizy danych podjęto się preprocesingu bazowego datasetu, który został w ten sposób przekształcony w dataset użytkowy. Podjęte kroki preprcessingu opisano w odpowiednich podsekcjach. Z racji, iż projekt z założenia wykorzystać miał metodę lasu losowego, to pominięto skalowanie danych oraz nie zdecydowano się na usunięcie silnie skorelowanych ze sobą cech. Odpowiedzialność za preprocessing datasetu bazowego ponosi moduł [`preprocess.py`](/src/data_preprocessing/preprocess.py).

Warto również wspomnieć, że rozważono łącznie 12 wariantów preprocessingu. Wszystkie z nich zostały przetestowane przy pomocy systemu walidacji modelu stworzonej na potrzeby projektu w celu wybrania najlepszego wairantu. Proces ten opisano w odpowiedniej podsekcji.

## 3.1 Wykonane operacje

### 3.1.1 Usunięcie wartości odstających

Rozkład zmiennej estymowanej w bazowym zbiorze danych jest znacząco przechylony na korzyść mniejszych (< 400,000 USD) wartości. W związku z tym rozważono możliwość ,,odcięcia'' z niego wartości skrajnych, odjstających (ang. *outliers*), które mogą negatywnie wpłynąć na jakoś predykcji.

Jako, że rozkład cechy estymowanej nie jesy wyśrodowkowany, to do ustalenia dolnego ($LB$) i górnego ($UB$) ograniczenia jej dozwolonych wartości, wykorzystano metodę *rozstępu ćwiartkowego* (ang. *interquartile range*, IQR). W owej metodzie wykorzystujemy 1 ($Q_1$) oraz 3 ($Q_3$) kwartyl rozkładu do obliczenia $UB$ oraz $LB$ zgodnie z poniższymi wzorami.

$$
LB = Q_1 - 1.5 \cdot IQR
$$

$$
UB = Q_3 + 1.5 \cdot IQR
$$

$$
IQR = Q_3 - Q_1
$$

Na potrzeby preprocessingu poza tymi klasycznymi wartościmi $LB$ i $UB$ sprawdzono także ich zmodyfikowany wariant obcinający mniej próbek, w których $IQR$ mnożony jest przez wartosc $2.0$. Co więcej nałożono dolny limit dla $LB$ na poziomie 0.00 USD, gdyż nie przewiduje się ujemnych wartości cechy estymowanej.

![histogram cechy estymowanej z oznaczonymi LB i UB](/datasets/plots/base/target_histogram_with_outliers.png)

### 3.1.2 Uzupełnienie wartości brakujących

Analiza datasetu bazowego wykazała również istnienie w nim wartości pustych, kótre powinny zostać uzupełnione w celu uzyskania lepszej jakości predykcji.

W przypadku wartości liczbowych zdecydowano się na uzupełnianie braków medianą wyliczoną z kolumny, w której pusta wartość się znajduje.

W przypadku wartości ketegorycznych rozważono dwie możliwości:

1. Uzupełnienie stałą wartością `NOT_PRESENT` z racji, że braki w tego typu kolumnach oznaczają brak pewnego rodzaju udognodnienia czy też cechy.
2. Uzupełnienie najczęściej występującą wartością z kolumny, w której pusta wartość się znajduje.

### 3.1.3 Stworzenie dodatkowych cech

W ramach preprocessingu dodano do datasetu **10 nowych cech**, stworzonych poprzez przetworzenie cech istniejących, które uznano za kluczowe z perspektywy użytkownika systemu.

1. `TotalSF`: Łączna powierzchnia nieruchomości.
2. `OverallQualityScore`: Iloczyn 10 stopniowej miary jakości materiałów użytych do budowy nieruchomości i 10 stopniowej miary jej stanu techniczego.
3. `HouseAge`: Wiek nieruchomości w latach w momencie sprzedaży.
4. `TotalBathrooms`: Łączna liczba łazienek w nieruchomości.
5. `GarageCapacity`: Miara pojemności garażu liczona w zaparkowanych autach.
6. `HasBasement`: Czy nieruchomośc ma piwnicę?
7. `HasSceondFloor`: Czy nierichomość jest wielopiętrowa?
8. `LotAreaLog`: Logarytm naturalny z powierzchni działki.
9. `QualityPriceInteraction`: Iloczyn 10 stopniowej miary jakości materiałów użytych do budowy nieruchomosci i powierzchni nieruchomości z pominięciem piwnicy.
10. `RecentRemodel`: Czy nieruchomość była poddana remontowi w przeciągu ostatnich 10 lat?

### 3.1.4 Zakodowanie zmiennych kategorycznych

Zmienne kategoryczne stanowią większośc bazowego zbioru danych. Co za tym idzie należło je zakodować, aby umożliwić modelowi ich efektywne wykorzystanie. W ramach tego korku rozważono 2 warianty tego typu kodowania.

1. Zakodowanie wszystkich kolumn kategorycznych przy pomocy *on hot encoding*. Co za tym idzie utrata części informacji dla tych kategorii, które towrzą ścisłą hierarchię.
2. Zakodowanie kolumn, gdzie kategorie tworzą ścisłą hierachię przy pomocy *ordinal encoding*. Reszta kategorii zakodowana przy pomocy *one hot encoding*. Co za tym idzie zachowanie informacji o hierachii dla kateogorii, gdzie taka występuje.

## 3.2 Analiza wariantów preprocessingu

W ramach pogłębionego preprocessingu rozważono 12 jego wariantów, które przedstawiono w tabeli poniżej. Zawarto w niej też wartości metryk $R^2$ oraz $MAPE$, które oceniają skuteczność modelu używającego każdego z wariantów preprocessingu.

| #  | Kodowanie kategorii | Usuwaj wartości odstające | Brakujące kategorie zmieniane na | $R^2$  | $MAPE$ |
|----|---------------------|---------------------------|----------------------------------|-------:|-------:|
| 1  | 1 HOT + ORDINAL     | spoza 1.5 IQR             | `NOT_PRESENT`                    | 0.8731 | 9.28 % |
| 2  | 1 HOT               | spoza 1.5 IQR             | `NOT_PRESENT`                    | 0.8704 | 9.38 % |
| 3  | 1 HOT + ORDINAL     | spoza 1.5 IQR             | `NOT_PRESENT`                    | 0.9039 | 8.44 % |
| 4  | 1 HOT               | spoza 2.0 IQR             | `NOT_PRESENT`                    | 0.9010 | 8.50 % |
| 5  | 1 HOT + ORDINAL     | nie usuwaj                | `NOT_PRESENT`                    | 0.8955 | 8.66 % |
| 6  | 1 HOT               | nie usuwaj                | `NOT_PRESENT`                    | 0.8923 | 9.22 % |
| 7  | 1 HOT + ORDINAL     | spoza 1.5 IQR             | dominanta                        | 0.8717 | 9.22 % |
| 8  | 1 HOT               | spoza 1.5 IQR             | dominanta                        | 0.8706 | 9.40 % |
| 9  | 1 HOT + ORDINAL     | spoza 1.5 IQR             | dominanta                        | 0.9049 | 8.37 % |
| 10 | 1 HOT               | spoza 2.0 IQR             | dominanta                        | 0.9043 | 8.52 % |
| 11 | 1 HOT + ORDINAL     | nie usuwaj                | dominanta                        | 0.8937 | 8.94 % |
| 12 | 1 HOT               | nie usuwaj                | dominanta                        | 0.8967 | 8.80 % |

Z przeproszdzonej analizy jasno wynika, że najkorzystniejsze wyniki estymacji uzyskujemy dla wariantu #9, który ostatecznie zotstał wykorzystany jako docelowy wariant preprocessingu w projekcie. Należy jednak zauważyć, że różnice pomiędzy wariantami nie są znaczące, nie przekraczając 1.13 p.p dla metryki $MAPE$ oraz 0.0343 dla metryki $R^2$.

[🏠 ← Powrót do README](../README.md)
