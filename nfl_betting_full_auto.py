import pandas as pd
import numpy as np
import nfl_data_py as nfl

# ---------- Config ----------
LATEST_AVAILABLE_SEASON = 2024  # fallback season for testing
INITIAL_EPA = 0.0

# ---------- Fetch Play-by-Play Data ----------
def fetch_pbp_data(season):
    """
    Fetch NFL play-by-play data for a given season.
    Falls back to latest historical season if the requested season is unavailable.
    """
    try:
        print(f"🌐 Fetching play-by-play data for the {season} season...")
        pbp_data = nfl.import_pbp_data([season])
        if pbp_data.empty:
            raise ValueError("Data not available for this season")
        return pbp_data
    except Exception as e:
        print(f"⚠️ Could not fetch data for {season}: {e}")
        print(f"➡️ Falling back to latest available season: {LATEST_AVAILABLE_SEASON}")
        pbp_data = nfl.import_pbp_data([LATEST_AVAILABLE_SEASON])
        return pbp_data

# ---------- Calculate EPA ----------
def calculate_epa(pbp_data):
    """
    Calculate Expected Points Added (EPA) for each play.
    Handles missing columns or empty datasets.
    """
    if pbp_data.empty:
        print("⚠️ PBP data is empty. Creating mock EPA column.")
        pbp_data['epa'] = INITIAL_EPA
        pbp_data['defteam'] = 'DEF'
        pbp_data['posteam'] = 'OFF'
        pbp_data['game_id'] = 'MOCK'
        return pbp_data

    if 'ep' not in pbp_data.columns or 'ep_before' not in pbp_data.columns:
        print("⚠️ Columns 'ep' or 'ep_before' missing. Creating mock EPA column.")
        pbp_data['epa'] = INITIAL_EPA
        if 'defteam' not in pbp_data.columns:
            pbp_data['defteam'] = 'DEF'
        if 'posteam' not in pbp_data.columns:
            pbp_data['posteam'] = 'OFF'
        if 'game_id' not in pbp_data.columns:
            pbp_data['game_id'] = 'MOCK'
        return pbp_data

    pbp_data['epa'] = pbp_data['ep'] - pbp_data['ep_before']
    return pbp_data

# ---------- Aggregate Net EPA by Team ----------
def aggregate_net_epa(pbp_data):
    """
    Aggregate both offensive and defensive EPA to compute net EPA for each team in each game.
    """
    # Offensive EPA
    off_epa = pbp_data.groupby(['game_id', 'posteam']).agg(
        off_total_epa=('epa', 'sum'),
        off_plays=('epa', 'count')
    ).reset_index()
    off_epa['off_epa_per_play'] = off_epa['off_total_epa'] / off_epa['off_plays']

    # Defensive EPA (negative EPA allowed by opponent)
    def_epa = pbp_data.groupby(['game_id', 'defteam']).agg(
        def_total_epa=('epa', 'sum'),
        def_plays=('epa', 'count')
    ).reset_index()
    def_epa['def_epa_per_play'] = -def_epa['def_total_epa'] / def_epa['def_plays']  # negative because it's defensive impact

    # Merge offensive and defensive EPA per team
    net_epa = pd.merge(off_epa, def_epa, left_on=['game_id', 'posteam'], right_on=['game_id', 'defteam'], how='left')
    net_epa['net_epa_per_play'] = net_epa['off_epa_per_play'] + net_epa['def_epa_per_play']

    # Keep relevant columns
    net_epa = net_epa[['game_id', 'posteam', 'off_epa_per_play', 'def_epa_per_play', 'net_epa_per_play']]
    return net_epa

# ---------- Generate Betting Recommendations ----------
def generate_betting_recommendations(net_epa):
    """
    Generate betting recommendations based on net EPA metrics.
    """
    df = net_epa.copy()
    if df.empty:
        print("⚠️ Net EPA data empty. Creating mock recommendations.")
        df = pd.DataFrame({
            'game_id': ['MOCK_1', 'MOCK_1'],
            'posteam': ['TeamA', 'TeamB'],
            'off_epa_per_play': [0.05, -0.02],
            'def_epa_per_play': [0.03, -0.01],
            'net_epa_per_play': [0.08, -0.03]
        })

    df['bet_favorite'] = df['net_epa_per_play'].apply(lambda x: 'Bet' if x > 0 else 'Avoid')
    df['confidence'] = df['net_epa_per_play'].abs()
    df['bet_type'] = df['confidence'].apply(lambda x: 'Spread' if x > 0.05 else 'No Bet')

    return df[['game_id', 'posteam', 'off_epa_per_play', 'def_epa_per_play', 'net_epa_per_play', 'bet_favorite', 'confidence', 'bet_type']]

# ---------- Main ----------
def main(season):
    pbp_data = fetch_pbp_data(season)
    pbp_data = calculate_epa(pbp_data)
    net_epa = aggregate_net_epa(pbp_data)
    betting_recommendations = generate_betting_recommendations(net_epa)

    print("\n📌 Betting Recommendations (Net EPA):")
    print(betting_recommendations.head(10))

# ---------- Entry Point ----------
if __name__ == "__main__":
    try:
        season_year = int(input("Enter season year (e.g., 2025): "))
    except ValueError:
        season_year = 2024
    main(season_year)


