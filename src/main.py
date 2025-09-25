import pandas as pd
from files.loading import load_raw_games_file
from files.cleaning import clean_data


def main():
    """Main function to load and display NBA games data."""
    df = load_raw_games_file()
    print(df.sample(10))
    print(df.columns)
    print(df.isnull().sum())
    df_cleaned = clean_data(df)
    print(df_cleaned.isnull().sum())

if __name__ == "__main__":
    main()
