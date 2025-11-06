
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import datetime as dt


def preprocess_data(df):
    df.drop(['LastPerformanceReview_Date', 'DaysLateLast30'],
            axis=1, inplace=True)
    df.dropna(thresh=2, inplace=True)
    df['DOB'] = pd.to_datetime(df['DOB'], format='%m/%d/%y')
    df['DateofTermination'] = pd.to_datetime(
        df['DateofTermination'], format='%m/%d/%y')
    df['DateofHire'] = pd.to_datetime(df['DateofHire'], format='%m/%d/%Y')
    return df


def get_employee_counts(df):
    return df.groupby('ManagerName')['EmpID'].count()


def analyze_manager_performance(df):
    df = df.copy()
    # Usunięcie wierszy z brakującymi danymi
    df = df.dropna(subset=['ManagerName', 'PerfScoreID'])

    # Słownik z nazwami ocen (ODWRÓCONY!)
    performance_names = {
        1.0: 'Fully Meets',  # Teraz liczby -> stringi
        2.0: 'Partially Meets',
        3.0: 'PIP',
        4.0: 'Exceeds',
        # Dodaj więcej, jeśli masz inne wartości PerfScoreID
    }

    # Zmiana wartości PerfScoreID na nazwy
    df['PerfScoreID'] = df['PerfScoreID'].map(performance_names)

    # Tabela krzyżowa z procentami (wiersze = 100%)
    crosstab_pct = pd.crosstab(
        df['ManagerName'], df['PerfScoreID'], normalize='index') * 100

    # Obliczenie średnich procentów dla każdej oceny (po kolumnach)
    average_percentages = crosstab_pct.mean(axis=0)

    # Obliczenie odchyleń od średniej
    deviations = crosstab_pct.subtract(average_percentages, axis=1)

    print("Odchylenia od średnich procentów:")
    print(deviations)

    # Wizualizacja odchyleń (heatmap)
    plt.figure(figsize=(14, 8))
    sns.heatmap(deviations, annot=True, cmap="coolwarm",
                fmt=".1f", center=0)  # coolwarm, center=0
    plt.title(
        "Odchylenia od średniego procentowego rozkładu PerfScoreID dla każdego managera")
    plt.xlabel("Ocena Wydajności")  # Zmieniony label
    plt.ylabel("ManagerName")
    plt.tight_layout()
    plt.show()


def analyze_manager_performance_chi2(df):
    """
    Analizuje związek pomiędzy ManagerName a PerfScoreID za pomocą testu Chi-Kwadrat i oblicza Cramera V.

    Args:
        df (pd.DataFrame): DataFrame zawierający dane.

    Returns:
        tuple: (chi2, p, dof, expected, cramer_v) - wyniki testu Chi-Kwadrat i Cramera V.
               Zwraca None, jeśli nie można wykonać testu.
    """

    # Usunięcie wierszy z brakującymi danymi w istotnych kolumnach
    df = df.dropna(subset=['ManagerName', 'PerfScoreID'])

    contingency_table = pd.crosstab(df['ManagerName'], df['PerfScoreID'])

    # Sprawdzenie, czy tabela kontyngencji ma oczekiwane liczebności >= 5
    min_expected = np.min(chi2_contingency(contingency_table)[3])
    if min_expected < 5:
        print("Ostrzeżenie: Niektóre oczekiwane liczebności w tabeli kontyngencji są mniejsze niż 5.")
        print("Wyniki testu Chi-Kwadrat mogą być niewiarygodne.")

    try:
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Obliczenie Cramera V
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        cramer_v = np.sqrt(phi2 / min((k - 1), (r - 1)))

        print("Chi-Kwadrat Test:")
        print(f"Chi2 Statistic: {chi2:.2f}")
        print(f"P-value: {p:.3f}")
        print(f"Degrees of Freedom: {dof}")
        print(f"Cramer's V: {cramer_v:.3f}")  # Dodane Cramera V
        print("Expected Frequencies Table:")
        print(expected)
        return chi2, p, dof, expected, cramer_v
    except ValueError as e:
        print(f"Błąd podczas wykonywania testu Chi-Kwadrat: {e}")
        print("Prawdopodobnie tabela kontyngencji zawiera puste wiersze lub kolumny.")
        return None, None, None, None, None


def count_seniority(row):

    if pd.isnull(row['DateofTermination']):
        end_date = dt.datetime(2019,9,27)
    else:
        end_date = row['DateofTermination']

    time_diff = end_date - row['DateofHire']

    return time_diff.days / 365.25

def add_seniority(df):
    df['Seniority'] = df.apply(lambda row: count_seniority(row),axis=1)
    return df

def analyze_recruitment_source_seniority(df):
    """
    Analizuje średni staż pracowników w zależności od źródła rekrutacji
    i wizualizuje wyniki.

    Args:
        df (pd.DataFrame): DataFrame zawierający dane, z kolumną 'Seniority'.
    """
    if 'Seniority' not in df.columns:
        print("Błąd: Kolumna 'Seniority' nie została znaleziona w DataFrame. Upewnij się, że funkcja add_seniority została wywołana.")
        return

    if 'RecruitmentSource' not in df.columns:
        print("Błąd: Kolumna 'RecruitmentSource' nie została znaleziona w DataFrame.")
        return

    # Usunięcie wierszy z brakującymi danymi w Recruitment Source lub Seniority
    df_cleaned = df.dropna(subset=['RecruitmentSource', 'Seniority'])

    if df_cleaned.empty:
        print("Brak danych do analizy stażu wg źródła rekrutacji po usunięciu brakujących wartości.")
        return

    # Grupujemy po źródle rekrutacji i obliczamy średni staż
    seniority_by_source = df_cleaned.groupby('RecruitmentSource')['Seniority'].mean()

    # Sortujemy wyniki malejąco
    seniority_by_source = seniority_by_source.sort_values(ascending=False)

    print("\nŚredni staż pracowników (w latach) wg źródła rekrutacji:")
    print(seniority_by_source)

    # Wizualizacja
    plt.figure(figsize=(14, 8))
    sns.barplot(x=seniority_by_source.index, y=seniority_by_source.values, palette='viridis')
    plt.title("Średni staż pracowników wg źródła rekrutacji", fontsize=16)
    plt.xlabel("Źródło Rekrutacji", fontsize=12)
    plt.ylabel("Średni Staż (lata)", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10) # Obrót etykiet dla czytelności
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()