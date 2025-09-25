import pandas as pd

def load_raw_games_file(filepath="data/nba_2008-2025_RAW.csv"):
    """Load the raw NBA games data from a CSV file."""
    return pd.read_csv(filepath)
