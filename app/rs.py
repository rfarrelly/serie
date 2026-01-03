import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------
def get_points(g_for, g_against):
    if g_for > g_against:
        return 3
    elif g_for == g_against:
        return 1
    else:
        return 0


def load_and_prepare_data(path):
    df = pd.read_csv(path)[["Date", "Home", "Away", "FTHG", "FTAG"]]
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    return df


def get_all_teams(df):
    return sorted(set(df["Home"]).union(set(df["Away"])))


# -------------------------
# Core Calculations
# -------------------------
def calculate_team_stats(games_so_far, all_teams):
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

    for _, row in games_so_far.iterrows():
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

    return stats


def calculate_team_ppgs(stats):
    team_ppgs = {}

    for team, s in stats.items():
        ppg = s["Pts"] / s["GP"] if s["GP"] > 0 else 0
        home_ppg = s["Home_Pts"] / s["Home_GP"] if s["Home_GP"] > 0 else 0
        away_ppg = s["Away_Pts"] / s["Away_GP"] if s["Away_GP"] > 0 else 0

        team_ppgs[team] = {
            "PPG": ppg,
            "Home_PPG": home_ppg,
            "Away_PPG": away_ppg,
        }

    return team_ppgs


def calculate_ppi_snapshot(games_so_far, team_ppgs, all_teams, current_date):
    records = []

    for team in all_teams:
        team_games = games_so_far[
            (games_so_far["Home"] == team) | (games_so_far["Away"] == team)
        ]

        opponent_strength_sum = 0
        opponent_count = 0

        for _, row in team_games.iterrows():
            if row["Home"] == team:
                opponent = row["Away"]
                opp_strength = team_ppgs[opponent]["Away_PPG"]
            else:
                opponent = row["Home"]
                opp_strength = team_ppgs[opponent]["Home_PPG"]

            opponent_strength_sum += opp_strength
            opponent_count += 1

        opp_ppg_avg = (
            opponent_strength_sum / opponent_count if opponent_count > 0 else 0
        )

        current_ppg = team_ppgs[team]["PPG"]
        ppi = round(current_ppg * opp_ppg_avg, 2)

        records.append(
            {
                "Date": current_date,
                "Team": team,
                "PPG": current_ppg,
                "Opponent_PPG": opp_ppg_avg,
                "PPI": ppi,
            }
        )

    return records


def build_ppi_dataframe(df, all_teams, apply_shift=True):
    ppi_records = []

    for current_date in df["Date"].unique():
        games_so_far = df[df["Date"] <= current_date]

        stats = calculate_team_stats(games_so_far, all_teams)
        team_ppgs = calculate_team_ppgs(stats)

        ppi_records.extend(
            calculate_ppi_snapshot(games_so_far, team_ppgs, all_teams, current_date)
        )

    ppi_df = pd.DataFrame(ppi_records)
    ppi_df.sort_values(["Date", "PPI"], ascending=[True, False], inplace=True)

    if apply_shift:
        cols_to_shift = ["PPG", "Opponent_PPG", "PPI"]
        ppi_df[cols_to_shift] = ppi_df.groupby("Team")[cols_to_shift].shift(1)

    return ppi_df.sort_values(["Team", "Date"])


# -------------------------
# Merge Back to Match Data
# -------------------------
def merge_ppi_into_matches(df, ppi_df):
    merged_df = df.merge(ppi_df, left_on=["Date", "Home"], right_on=["Date", "Team"])
    merged_df = merged_df.merge(
        ppi_df, left_on=["Date", "Away"], right_on=["Date", "Team"]
    )

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

    return merged_df[
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


# -------------------------
# Execution
# -------------------------
def compute_ppi(data: pd.DataFrame, shift: bool = False):
    df = load_and_prepare_data(data)
    all_teams = get_all_teams(df)
    ppi_df = build_ppi_dataframe(df, all_teams, apply_shift=shift)
    return merge_ppi_into_matches(df, ppi_df).dropna(how="any", axis="index")


ppi_df = compute_ppi(
    data="DATA/FBREF/National-League/National-League_2025-2026.csv", shift=True
)
