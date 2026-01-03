import numpy as np
import pandas as pd

df = pd.read_csv("DATA/FBREF/National-League/National-League_2025-2026.csv")[
    ["Date", "Home", "Away", "FTHG", "FTAG"]
]
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)


# Helper for points
def get_points(g_for, g_against):
    if g_for > g_against:
        return 3
    elif g_for == g_against:
        return 1
    else:
        return 0


# Unique Teams
all_teams = sorted(list(set(df["Home"].unique()).union(set(df["Away"].unique()))))

# Store results
ppi_records = []

# Iterate through each unique date to build the time series
unique_dates = df["Date"].unique()

for current_date in unique_dates:
    # 1. Snapshot: Get all games played up to and including this date
    games_so_far = df[df["Date"] <= current_date].copy()

    # 2. Calculate Team Stats (Games, Points, Home Games, Home Points, Away Games, Away Points)
    # We will compute this efficiently for the current snapshot

    stats = {
        team: {
            "GP": 0,
            "Pts": 0,
            "Home_GP": 0,
            "Home_Pts": 0,
            "Away_GP": 0,
            "Away_Pts": 0,
        }
        for team in all_teams
    }

    for idx, row in games_so_far.iterrows():
        h, a = row["Home"], row["Away"]
        hg, ag = row["FTHG"], row["FTAG"]

        hp = get_points(hg, ag)
        ap = get_points(ag, hg)

        stats[h]["GP"] += 1
        stats[h]["Pts"] += hp
        stats[h]["Home_GP"] += 1
        stats[h]["Home_Pts"] += hp

        stats[a]["GP"] += 1
        stats[a]["Pts"] += ap
        stats[a]["Away_GP"] += 1
        stats[a]["Away_Pts"] += ap

    # 3. Calculate PPGs for the snapshot
    team_ppgs = {}
    for team, s in stats.items():
        # Overall PPG
        ppg = s["Pts"] / s["GP"] if s["GP"] > 0 else 0
        # Home PPG
        home_ppg = s["Home_Pts"] / s["Home_GP"] if s["Home_GP"] > 0 else 0
        # Away PPG
        away_ppg = s["Away_Pts"] / s["Away_GP"] if s["Away_GP"] > 0 else 0

        team_ppgs[team] = {"PPG": ppg, "Home_PPG": home_ppg, "Away_PPG": away_ppg}

    # 4. Calculate Opponent Strength and PPI for each team
    for team in all_teams:
        # Get team's games
        team_games = games_so_far[
            (games_so_far["Home"] == team) | (games_so_far["Away"] == team)
        ]

        opponent_strength_sum = 0
        opponent_count = 0

        for idx, row in team_games.iterrows():
            if row["Home"] == team:
                opponent = row["Away"]
                # Played at Home -> Get Opponent's Away PPG
                opp_strength = team_ppgs[opponent]["Away_PPG"]
            else:
                opponent = row["Home"]
                # Played Away -> Get Opponent's Home PPG
                opp_strength = team_ppgs[opponent]["Home_PPG"]

            opponent_strength_sum += opp_strength
            opponent_count += 1

        opp_ppg_avg = (
            opponent_strength_sum / opponent_count if opponent_count > 0 else 0
        )

        # PPI Calculation
        current_ppg = team_ppgs[team]["PPG"]
        ppi = round(current_ppg * opp_ppg_avg, 2)

        ppi_records.append(
            {
                "Date": current_date,
                "Team": team,
                "PPG": current_ppg,
                "Opponent_PPG": opp_ppg_avg,
                "PPI": ppi,
            }
        )

# Create DataFrame
ppi_df = pd.DataFrame(ppi_records)
ppi_df.sort_values(["Date", "PPI"], ascending=[True, False], inplace=True)

cols_to_shift = ["PPG", "Opponent_PPG", "PPI"]

# Group by 'Team' and shift the selected columns down by 1
ppi_df[cols_to_shift] = ppi_df.groupby("Team")[cols_to_shift].shift(1)

# Resulting DataFrame (Sorted by Team and Date for better visibility)
ppi_df = ppi_df.sort_values(["Team", "Date"])

merged_df = df.merge(ppi_df, left_on=["Date", "Home"], right_on=["Date", "Team"])
merged_df = merged_df.merge(ppi_df, left_on=["Date", "Away"], right_on=["Date", "Team"])

merged_df = merged_df.rename(
    columns={
        "PPG_x": "HomeTeamTotalPPG",
        "PPG_y": "AwayTeamTotalPPG",
        "Opponent_PPG_x": "HomeTeamOpponentPPG",
        "Opponent_PPG_y": "AwayTeamOpponentPPG",
        "PPI_x": "HomeTeamPPI",
        "PPI_y": "AwayTeamPPI",
    }
)

merged_df = merged_df[
    [
        "Date",
        "Home",
        "Away",
        "FTHG",
        "FTAG",
        "HomeTeamTotalPPG",
        "AwayTeamTotalPPG",
        "HomeTeamOpponentPPG",
        "AwayTeamOpponentPPG",
        "HomeTeamPPI",
        "AwayTeamPPI",
    ]
]
