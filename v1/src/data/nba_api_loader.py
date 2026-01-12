from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

def load_api_data(seasons):
    """
    Pulling data from the nba_api for each season from 2008 to 2025 and saving it as a csv.
    2008 is the first season with spread and total data.
    
    Parameters:
    -----------
    seasons : list
        List of season years (e.g., [2008, 2009, ..., 2025])
    add_possessions : bool
        Whether to fetch and add possession data (default: True)
    api_delay : float
        Delay between API calls in seconds (default: 0.6)
    """
    for season in seasons:
        filename = Path(f"../data/raw/nba_api_{season}_RAW.csv")
        
        if not filename.exists():
            print(f"\nProcessing season {season}...")
            
            # Create season tag
            if season % 100 < 10:
                season_tag = f'{season-1}-0{season % 100}'
            else: 
                season_tag = f'{season-1}-{season % 100}'
            
            # Fetch basic game data
            print(f"  Fetching game data for {season_tag}...")
            season_api = leaguegamefinder.LeagueGameFinder(
                season_nullable=season_tag, 
                league_id_nullable='00'
            )
            df = season_api.get_data_frames()[0]        
            df.columns = df.columns.str.lower()
            
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"  Saved to {filename}")
        else:
            # File exists - check if possessions column needs to be added
            print(f"\nSeason {season} file already exists...")

def merge_api_dfs(seasons):
    api_dfs = []
    for season in seasons:
        api_df = pd.read_csv(Path(f"../../data/raw/nba_api_{season}_RAW.csv"))
        api_dfs.append(api_df)

    full_api_df = pd.concat(api_dfs, ignore_index=True)
    full_api_df.to_csv(Path("../../data/raw/nba_ALL_SEASONS_API_RAW.csv"), index=False)