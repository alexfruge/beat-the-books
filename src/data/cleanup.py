import pandas as pd
from pathlib import Path
from utils import team_map
from data.nba_api_loader import load_api_data

def create_csvs(df: pd.DataFrame):
    seasons = df['season'].unique()
    for season in seasons:
        season_df = df[df['season'] == season]
        season_df.columns = season_df.columns.str.lower()
        season_df.to_csv(Path(f"../data/interim/nba_{season}_CLEANED.csv"), index=False)

    # as well as a cleaned version of the full dataset
    df.to_csv(Path("../data/interim/nba_ALL_SEASONS_CLEANED.csv"), index=False)

def clean_API_data(df):
    """
    Cleans up discrepancies between data pulled from .csv and nba_api.
    In general, we match to the format of the .csv data.
    """
    df = df.copy()

    # remove null entries
    df = df.dropna(how='any')
    
    # need to change their abbreviations
    df['team_abbreviation'] = df['team_abbreviation'].replace({'NJN': 'BKN', 'NOH': 'NOP', 'SEA': 'OKC', 'CHO': 'CHA'})
    # lets make the abreviations uppercase for consistency (just in case lowercase ones show up)
    df['team_abbreviation'] = df['team_abbreviation'].str.upper()

    # make team names match value from team_map
    df['team_name'] = df['team_abbreviation'].map(team_map())

    # drop any rows that don't have a valid team abbreviation
    df = df.dropna(subset=['team_abbreviation', 'team_name'])

    # change some column names for consistency with the other dataset
    df = df.rename(columns={'season_id': 'season', 'PTS': 'points', 'game_date': 'date'})

    # filtering to only actual games between two teams (no all-star/preseason games)
    df = df[df['team_id'].astype(str).str.contains('127')] # the 127 is a common value found for actual teams, not All-star/preseason games
    df = df[df['game_id'].duplicated(keep=False)]   # since each line corresponds to a single team, we remove all of the entries that don't have a matching team
                                                                # this is mostly cleanup from all games that have a team whose ID doesn't contain '127'                                                    

    df = df.sort_values(by='date') # sorting for consistency with CSV data

    df = df.drop(columns=['team_id', 'team_name', 'min', 'plus_minus'])

    return df

def merge_api_dfs(seasons):
    api_dfs = []
    for season in seasons:
        api_df = pd.read_csv(Path(f"../data/raw/nba_api_{season}_RAW.csv"))
        api_dfs.append(api_df)

    full_api_df = pd.concat(api_dfs, ignore_index=True)
    full_api_df.to_csv(Path("../data/raw/nba_ALL_SEASONS_API_RAW.csv"), index=False)

def format_csv_data(seasons):
    merged_dfs = []
    for season in seasons:
        csv_df = pd.read_csv(Path(f"../data/interim/nba_{season}_CLEANED.csv"))

        # not much to do aside from changing the case of the team abbreviations
        csv_df['away'] = csv_df['away'].str.upper()
        csv_df['home'] = csv_df['home'].str.upper()

        csv_df['away'] = csv_df['away'].replace({'NO': 'NOP', 'SA': 'SAS', 'NY': 'NYK', 'GS': 'GSW', 'UTAH': 'UTA', "WSH": "WAS"})
        csv_df['home'] = csv_df['home'].replace({'NO': 'NOP', 'SA': 'SAS', 'NY': 'NYK', 'GS': 'GSW', 'UTAH': 'UTA', "WSH": "WAS"})

        csv_df.to_csv(Path(f"../data/interim/nba_{season}_CLEANED.csv"), index=False)
        merged_dfs.append(csv_df)

    full_merged_df = pd.concat(merged_dfs, ignore_index=True)
    full_merged_df.to_csv(Path("../data/interim/nba_ALL_SEASONS_CLEANED.csv"), index=False)

def expand_csv_df():
    # expanding the csv data to have a row for each team in each game, mirroring the format of the API data
    full_csv_df = pd.read_csv(Path("../data/interim/nba_ALL_SEASONS_CLEANED.csv"))

    # Create a new DataFrame with duplicated entries for home_team = 0 and home_team = 1
    df_home = full_csv_df.copy()
    df_home['home_team'] = 1

    df_away = full_csv_df.copy()
    df_away['home_team'] = 0

    full_csv_df_expanded = pd.concat([df_home, df_away], ignore_index=True)

    full_csv_df_expanded['team_abbreviation'] = full_csv_df_expanded.apply(lambda row: row['home'] if row['home_team'] == 1 else row['away'], axis=1)
    full_csv_df_expanded['points'] = full_csv_df_expanded.apply(lambda row: row['score_home'] if row['home_team'] == 1 else row['score_away'], axis=1)
    full_csv_df_expanded = full_csv_df_expanded.sort_values(by=['date', 'home', 'team_abbreviation'])

    full_csv_df_expanded = full_csv_df_expanded.drop(columns=['home', 'away', 'score_home', 'score_away']).sort_values(by=['date', 'home_team', 'team_abbreviation'])

    full_csv_df_expanded.to_csv(Path("../data/interim/nba_ALL_SEASONS_CLEANED_EXPANDED.csv"), index=False)

def expand_api_df():
    # some cleanup and expansion of the API data
    full_api_df = pd.read_csv(Path("../data/interim/nba_ALL_SEASONS_API_CLEANED.csv"))

    full_api_df['home_team'] = full_api_df['matchup'].str.contains('vs').astype(int)

    full_api_df = full_api_df.drop(columns=['matchup'])

    full_api_df.to_csv(Path("../data/interim/nba_ALL_SEASONS_API_CLEANED_EXPANDED.csv"), index=False)

def match_games(api_df, csv_df):
    """
    Matches games between api_df and csv_df based on 'date' and 'team_abbreviation'.
    Returns a merged DataFrame with suffixes to distinguish columns from each source.
    """
    merged = pd.merge(
        api_df,
        csv_df,
        left_on=['date', 'team_abbreviation'],
        right_on=['date', 'team_abbreviation'],
        suffixes=('_api', '_csv'),
        how='inner'
    )
    merged = merged.rename(columns={'home_team_api': 'home_team'})
    merged = merged.drop(columns=['season_api', 'season_csv', 'home_team_csv', 'points'])
    merged = merged.sort_values(by=['game_id'])
    merged.to_csv(Path("../data/processed/nba_ALL_SEASONS_MATCHED.csv"), index=False)
    return merged

def main():
    csv_df = pd.read_csv(Path("../data/raw/nba_2008-2025_RAW.csv"))
    csv_df = csv_df.drop(["moneyline_home", "moneyline_away", "h2_spread", "h2_total"], axis=1, errors='ignore')
    csv_df = csv_df.dropna(how='any')

    create_csvs(csv_df)

    seasons = csv_df['season'].unique()

    # API data
    if all(Path(f"../data/raw/nba_api_{season}_RAW.csv").exists() for season in seasons):
        print("All individual season API data files exist. Combining into one file.")
    else:
        load_api_data(seasons)

    merge_api_dfs(seasons)

    for season in seasons:
        api_df = pd.read_csv(Path(f"../data/raw/nba_api_{season}_RAW.csv")) 
        print(api_df)
        api_df = clean_API_data(api_df)
        api_df.to_csv(Path(f"../data/interim/nba_api_{season}_CLEANED.csv"), index=False)

    full_api_df = pd.read_csv(Path("../data/raw/nba_ALL_SEASONS_API_RAW.csv"))
    full_api_df = clean_API_data(full_api_df)
    full_api_df.to_csv(Path("../data/interim/nba_ALL_SEASONS_API_CLEANED.csv"), index=False)
    
    # some formatting for csv data
    format_csv_data(seasons)

    # some cleanup and expansion of API data
    full_api_df['home_team'] = full_api_df['matchup'].str.contains('vs').astype(int)
    full_api_df = full_api_df.drop(columns=['matchup'])
    full_api_df.to_csv(Path("../data/interim/nba_ALL_SEASONS_API_CLEANED_EXPANDED.csv"), index=False)
    
    expand_csv_df()
    expand_api_df()

    full_csv_df = pd.read_csv(Path("../data/interim/nba_ALL_SEASONS_CLEANED_EXPANDED.csv"))

    matched_df = match_games(full_api_df, full_csv_df)


if __name__ == "__main__":
    main()