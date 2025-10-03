import pandas as pd
from pathlib import Path

from data.nba_api_loader import load_api_data
from data import cleanup

def preprocessing():
    print('penits')

def main():
    """Main function to load and display NBA games data."""
    if not Path('../data/processed/nba_ALL_SEASONS_MATCHED.csv').exists():
        load_api_data(list(range(2008, 2026)))
    
    cleanup.main()

if __name__ == "__main__":
    main()
