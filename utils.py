
import pandas as pd


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
