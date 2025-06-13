
[🏠 ← Powrót do README](../README.md)

---

# 6. Wyniki i wnioski

## 6.1 Podsumowanie osiągnięć projektu

Projekt wykazał skuteczność wykorzystania metody lasu losowego (Random Forest) do predykcji cen nieruchomości na podstawie zbioru danych z Ames (Iowa). Przeprowadzona analiza cech, solidny preprocessing, oraz ewaluacja 12 wariantów przygotowania danych umożliwiły uzyskanie wysokich wartości miar oceny modelu.

### Kluczowe metryki
Najlepszy wariant preprocessingu (#9 z tabeli w [analizie wariantów](03%20-%20preprocessing.md#32-analiza-wariantów-preprocessingu)) osiągnął:

- **R² = 0.9049** - model wyjaśnia 90.5% wariancji cen nieruchomości
- **MAPE = 8.37%** - średni błąd procentowy poniżej 10%

Wyniki te potwierdzają, że model dobrze oddaje relacje między cechami nieruchomości a ich cenami, a uzyskana dokładność predykcji jest wystarczająco wysoka, by mogła być użyteczna w praktyce – zarówno dla inwestorów, jak i klientów indywidualnych.

## 6.2 Znaczenie inżynierii cech

Dodatkowe cechy stworzone w ramach preprocessingu okazały się trafnymi agregatami informacji i istotnie wpłynęły na skuteczność modelu:

- **TotalSF** - określające całkowitą powierzchnię nieruchomości
- **OverallQualityScore** - iloczyn jakości wykonania oraz stanu technicznego  
- **RecentRemodel** - określające czy nieruchomość była remontowana w ciągu ostatnich 10 lat

Trafne zakodowanie zmiennych kategorycznych (poprzez połączenie One-Hot oraz Ordinal Encoding) także przyczyniło się do wysokiej jakości predykcji.

## 6.3 Możliwości dalszego rozwoju

### 6.3.1 Rozszerzenie geograficzne

Przede wszystkim warto rozważyć rozszerzenie zbioru danych o nieruchomości z innych lokalizacji, co pozwoliłoby zwiększyć uniwersalność modelu oraz uczynić go bardziej przydatnym w różnych kontekstach geograficznych. Obecny model został stworzony na podstawie danych z miasta Ames w stanie Iowa, co ogranicza jego zastosowanie wyłącznie do tego regionu.

#### Wyzwania metodologiczne
Jednak rozszerzenie geograficzne niesie ze sobą znaczące wyzwania metodologiczne:

- **Różnice regionalne**: Rynki nieruchomości w różnych lokalizacjach charakteryzują się często fundamentalnie odmiennymi czynnikami
- **Pokrycie cech**: Kluczowym wyzwaniem byłoby zapewnienie odpowiedniego pokrycia cech między zbiorami danych z różnych regionów
- **Specyfika lokalna**: Dataset z nowych lokalizacji musiałby zawierać nie tylko podobne cechy strukturalne (powierzchnia, liczba pokoi, wiek budynku), ale także uwzględniać specyfikę danego rynku mieszkaniowego

### 6.3.2 Alternatywne algorytmy uczenia maszynowego

Kolejnym naturalnym krokiem jest przetestowanie alternatywnych algorytmów uczenia maszynowego. Choć Random Forest zapewnił wysoką jakość predykcji, istnieje możliwość dalszej poprawy wyników poprzez zastosowanie takich metod jak:

- **XGBoost** - gradient boosting z zaawansowaną regularyzacją
- **LightGBM** - szybki gradient boosting
- **Sieci neuronowe (MLP)** - dla złożonych nieliniowych zależności
- **Modele liniowe typu ElasticNet** - dla większej interpretowalności

Porównanie ich skuteczności względem obecnego rozwiązania może przynieść bardziej efektywny lub lepiej skalujący się model.

### 6.3.3 Zaawansowana analiza cech

Warto rozważyć zastosowanie zaawansowanych metod selekcji cech, takich jak:

- **Recursive Feature Elimination** - automatyczna selekcja najważniejszych cech
- **Analiza SHAP** - wyjaśnienie wpływu poszczególnych cech na predykcje
- **LIME** - lokalne wyjaśnienia dla pojedynczych predykcji

Tego typu narzędzia mogą nie tylko poprawić jakość predykcji, ale również zwiększyć przejrzystość działania modelu. Pozwalają lepiej zrozumieć, które cechy mają największy wpływ na wycenę nieruchomości, co może być cenne z punktu widzenia użytkownika końcowego.

## 6.4 Ulepszenia interfejsu użytkownika

### 6.4.1 Problemy z obecną aplikacją

Obecna aplikacja Streamlit, choć funkcjonalna, ma znaczące możliwości poprawy w zakresie UX:

**Problem z formatem dzielnic**: Dropdown "Neighborhood" wyświetla skróty takie jak:
- `"Blmngtn"` zamiast "Bloomington Heights"
- `"NAmes"` zamiast "North Ames"  
- `"NoRidge"` zamiast "Northridge"

Te skróty są niezrozumiałe dla przeciętnego użytkownika i mogą ograniczać użyteczność aplikacji.

Ponadto, użyto amerykańskich jednostek miary dostępnych w datasecie źródłowym, przez co użytkownicy ich nie znający mogą mieć problemy z wprowadzaniem danych o odpowiedniej wartości.

### 6.4.2 Proponowane usprawnienia

**Walidacja danych**: Dodanie walidacji wprowadzanych danych, tak aby użytkownik nie mógł zażądać estymacji ceny domu o np.:
- Najmniejszej możliwej powierzchni użytkowej z 5 łazienkami
- Garażu o powierzchni stanowiącej wielokrotność powierzchni samego domu
- Innych nierealistycznych kombinacji cech

Ze względu na absurdalność takiego połączenia cech.

## 6.5 Funkcjonalności dla użytkowników

### 6.5.1 Predykcje grupowe

Kolejną przydatną funkcją mogłoby być wprowadzenie możliwości grupowej predykcji cen nieruchomości na podstawie pliku wejściowego (np. CSV). Byłoby to szczególnie użyteczne dla:

- **Agencji nieruchomości** - masowa wycena portfela
- **Deweloperów** - analiza większych zestawów danych
- **Inwestorów** - screening możliwości inwestycyjnych

### 6.5.2 Mechanizmy wyjaśniające

Rozszerzenie projektu o mechanizmy wyjaśniające działanie modelu zwiększyłoby jego transparentność i zaufanie użytkowników poprzez:

- **SHAP values** - globalne i lokalne wyjaśnienia
- **LIME** - intuicyjne wyjaśnienia dla pojedynczych przypadków
- **Feature importance** - ranking najważniejszych cech

## 6.6 Ocena końcowa projektu

### 6.6.1 Kluczowe sukcesy

Projekt stanowi udaną implementację nowoczesnych technik machine learning w praktycznym zastosowaniu. Osiągnięte wyniki **(R² = 0.905, MAPE = 8.37%)** plasują rozwiązanie na poziomie komercyjnych narzędzi wyceny nieruchomości.

**Główne osiągnięcia**:
- ✅ **Stworzenie dokładnego i interpretowalnego modelu**
- ✅ **Kompleksowa metodologia od danych do aplikacji**  
- ✅ **Praktyczne narzędzie z przyjaznym interfejsem**

### 6.6.2 Najważniejsze wnioski z projektu

1. **Jakość danych jest kluczowa dla sukcesu modelu** - starannie przeprowadzony preprocessing miał większy wpływ na wyniki niż dobór algorytmu

2. **Inżynieria cech ma większy wpływ niż wybór algorytmu** - stworzenie 10 dodatkowych, przemyślanych cech znacząco poprawiło dokładność modelu

3. **Interpretowalność jest równie ważna jak dokładność** - możliwość wyjaśnienia predykcji zwiększa zaufanie użytkowników i praktyczną użyteczność modelu

### 6.6.3 Perspektywy rozwoju

Projekt ma solidne fundamenty do rozszerzenia w kierunku kompleksowej platformy analitycznej dla rynku nieruchomości. Możliwość skalowania geograficznego i funkcjonalnego otwiera ścieżki do komercyjnego wykorzystania oraz dalszych badań naukowych w obszarze automatycznej wyceny nieruchomości.

---


[🏠 ← Powrót do README](../README.md)
