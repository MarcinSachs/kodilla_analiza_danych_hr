
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
        end_date = dt.datetime(2019, 9, 27)
    else:
        end_date = row['DateofTermination']

    time_diff = end_date - row['DateofHire']

    return time_diff.days / 365.25


def add_seniority(df):
    df['Seniority'] = df.apply(lambda row: count_seniority(row), axis=1)
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
    seniority_by_source = df_cleaned.groupby('RecruitmentSource')[
        'Seniority'].mean()

    # Sortujemy wyniki malejąco
    seniority_by_source = seniority_by_source.sort_values(ascending=False)

    print("\nŚredni staż pracowników (w latach) wg źródła rekrutacji:")
    print(seniority_by_source)

    # Wizualizacja
    plt.figure(figsize=(14, 8))
    sns.barplot(x=seniority_by_source.index,
                y=seniority_by_source.values, palette='viridis')
    plt.title("Średni staż pracowników wg źródła rekrutacji", fontsize=16)
    plt.xlabel("Źródło Rekrutacji", fontsize=12)
    plt.ylabel("Średni Staż (lata)", fontsize=12)
    # Obrót etykiet dla czytelności
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def analyze_martial_status_satisfaction_corelation(df):
    """
    Analizuje związek między stanem cywilnym a zadowoleniem z pracy.
    Wyświetla surowe liczby, procentowy rozkład i wizualizuje wyniki heatmapą.
    """
    # Usunięcie wierszy z brakującymi danymi w MaritalDesc lub EmpSatisfaction
    df_cleaned = df.dropna(subset=['MaritalDesc', 'EmpSatisfaction'])
    if df_cleaned.empty:
        print("Brak danych do analizy po usunięciu brakujących wartości dla stanu cywilnego i satysfakcji.")
        return

    # 1. Wyświetlenie surowych liczebności
    satisfaction_counts = df_cleaned.groupby(
        'MaritalDesc')['EmpSatisfaction'].value_counts().sort_index()
    print("\n--- Surowe liczby pracowników dla każdego poziomu satysfakcji w zależności od stanu cywilnego ---")
    print(satisfaction_counts)

    # 2. Obliczenie procentowego rozkładu satysfakcji w obrębie każdej grupy stanu cywilnego
    # 'normalize=True' oblicza proporcje w obrębie grupy (na wiersz)
    satisfaction_percentages = df_cleaned.groupby('MaritalDesc')[
        'EmpSatisfaction'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
    # Posortowanie kolumn (poziomów satysfakcji) dla lepszej czytelności
    satisfaction_percentages = satisfaction_percentages.reindex(
        columns=sorted(satisfaction_percentages.columns))

    print("\n--- Procentowy rozkład poziomów satysfakcji z pracy w zależności od stanu cywilnego (wiersze sumują się do 100%) ---")
    print(satisfaction_percentages.round(2))

    # 3. Wizualizacja procentowego rozkładu za pomocą heatmapy
    plt.figure(figsize=(12, 7))
    sns.heatmap(satisfaction_percentages, annot=True, fmt=".1f",
                cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Procent'})
    plt.title(
        'Procentowy rozkład satysfakcji z pracy w zależności od stanu cywilnego', fontsize=16)
    plt.xlabel(
        'Poziom Satysfakcji z Pracy (1=bardzo niska, 5=bardzo wysoka)', fontsize=12)
    plt.ylabel('Stan Cywilny', fontsize=12)
    plt.tight_layout()
    plt.show()


def get_still_working_employees(df):
    df = df.copy()
    df = df.loc[df['EmploymentStatus'] == 'Active']
    return df


def calculate_age(born, reference_date):
    if born.year > reference_date.year:  # Sprawdzanie czy data urodzenia nie jest z przyszłości
        return None
    return reference_date.year - born.year - ((reference_date.month, reference_date.day) < (born.month, born.day))


def add_age(df, reference_date_str='2019-09-27'):
    """Dodaje kolumnę 'Age' do DataFrame, obliczoną względem daty odniesienia.

    Args:
        df (pd.DataFrame): DataFrame zawierający dane pracowników.
        reference_date_str (str): Data w formacie 'RRRR-MM-DD'. Domyślnie '2019-09-27'.
    """
    reference_date = pd.to_datetime(
        reference_date_str).date()  # Konwersja na obiekt date
    df['Age'] = df.apply(lambda row: calculate_age(
        row['DOB'], reference_date), axis=1)
    # Usunięcie brakujących wartości wieku
    df = df.dropna(subset=['Age'])
    return df


def analyse_age_still_working_employees(df):
    average_age = df['Age'].mean()
    youngest_age = df['Age'].min()
    oldest_age = df['Age'].max()
    print(f"Średni wiek pracowników: {average_age:.2f} lat")
    print(f"Najmłodszy pracownik ma {youngest_age} lat")
    print(f"Najstarszy pracownik ma {oldest_age} lat")

    print(df.groupby('Age')['EmpID'].count())

    # Wizualizacja
    plt.figure(figsize=(14, 8))
    ax = sns.countplot(x='Age', data=df,
                       palette='viridis')  # Zapisujemy obiekt Axes

    # Pobranie unikalnych, posortowanych wieków używanych na wykresie
    unique_ages = sorted(df['Age'].unique())

    # Konwersja wartości wieku na indeksy na osi X
    average_age_index = unique_ages.index(round(average_age)) if round(
        average_age) in unique_ages else None  # zaokraglam wiek do int
    youngest_age_index = unique_ages.index(youngest_age)
    oldest_age_index = unique_ages.index(oldest_age)

    plt.title('Liczba pracowników w zależności od wieku', fontsize=16)
    plt.xlabel('Wiek', fontsize=12)
    plt.ylabel('Liczba pracowników', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Dodanie linii pionowych dla średniego, minimalnego i maksymalnego wieku
    if average_age_index is not None:
        plt.axvline(x=average_age_index, color='red', linestyle='--',
                    label=f'Średni wiek: {average_age:.2f}')
    plt.axvline(x=youngest_age_index, color='green',
                linestyle='-', label=f'Najmłodszy: {youngest_age}')
    plt.axvline(x=oldest_age_index, color='blue', linestyle='-',
                label=f'Najstarszy: {oldest_age}')

    plt.legend()
    plt.tight_layout()
    plt.show()


def analyse_project_count_by_age(df):
    # Podział na kategorie wiekowe
    bins = [27, 35, 42, 50]
    labels = ['Młoda (27-34)', 'Średnia (35-41)', 'Starsza (42-50)']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    # Obliczenie średniej liczby projektów dla każdej grupy wiekowej
    average_projects_by_age = df.groupby(
        'AgeGroup')['SpecialProjectsCount'].mean()
    print("\nŚrednia liczba projektów w zależności od grupy wiekowej:")
    print(average_projects_by_age)

    # Wizualizacja wyników
    plt.figure(figsize=(10, 6))
    sns.barplot(x=average_projects_by_age.index,
                y=average_projects_by_age.values, palette='viridis')
    plt.title('Średnia liczba projektów w zależności od grupy wiekowej')
    plt.xlabel('Grupa Wiekowa')
    plt.ylabel('Średnia liczba projektów')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
