import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import utils


def main():
    df = pd.read_csv('HRDataset.csv')
    df = utils.preprocess_data(df)
    df.head()
    sns.set_style('darkgrid')
    df.pivot_table(index=df['PerfScoreID'], columns=df['ManagerID'], values='EmpID',aggfunc='count').fillna(0)

if __name__ == "__main__":
    main()
