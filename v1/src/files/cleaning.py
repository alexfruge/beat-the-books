import pandas as pd

def clean_data(df):
    """
    Clean the NBA games data by handling missing values and duplicates.
    Also removes columns that we won't be using for analysis. (mainly moneyline columns)
    """
    df = df.drop(["moneyline_home", "moneyline_away"], axis=1, errors='ignore')
    df = df.dropna(how='any')
    return df