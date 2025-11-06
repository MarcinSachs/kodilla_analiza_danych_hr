import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import utils


def main():
    df = pd.read_csv('HRDataset.csv')
    df = utils.preprocess_data(df)

    print(utils.get_employee_counts(df))
    # ręczna analiza
    utils.analyze_manager_performance(df)
    # test chi2
    chi2_results = utils.analyze_manager_performance_chi2(df)

    # Sprawdzenie czy p-value nie jest None
    if chi2_results and chi2_results[1] is not None:
        chi2, p, dof, expected, cramer_v = chi2_results
        alpha = 0.05  # Poziom istotności
        if p < alpha:
            print("Odrzucamy hipotezę zerową. Istnieje statystycznie istotna zależność między ManagerName a PerfScoreID.")
        else:
            print("Nie ma podstaw do odrzucenia hipotezy zerowej. Brak statystycznie istotnej zależności między ManagerName a PerfScoreID.")
    # Dodanie kolumny z dugocia stazu
    df = utils.add_seniority(df)

    # Analiza stażu
    utils.analyze_recruitment_source_seniority(df)



if __name__ == "__main__":
    main()
