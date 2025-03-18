import stats
import os
import pandas as pd
from collections import Counter
from pathlib import Path

import config, ingestion, plotting

BASE_URL = "https://fbref.com/en/comps"
LEAGUE_CONFIG = config.League.ENL
LEAGUE_NAME = LEAGUE_CONFIG.fbref_name
SEASON = "2024-2025"
DATA_DIRECTORY = f"./DATA/FBREF/{LEAGUE_NAME}"
HISTORICAL_DATA_FILE_NAME = f"{LEAGUE_NAME}_{SEASON}.csv"
FUTURE_FIXTURES_FILE_NAME = f"unplayed_{LEAGUE_NAME}_{SEASON}.csv"
RPI_PLOTS_SAVE_DIRECORY = f"./PLOTS/{LEAGUE_NAME}_{SEASON}/rpi"
PPG_PLOTS_SAVE_DIRECORY = f"./PLOTS/{LEAGUE_NAME}_{SEASON}/ppg"
WEEKS = [23, 26, 28, 30]
WINDOW = 1
RPI_DIFF_THRESHOLD = 0.1


def get_data(save_path: str, season: str = "current"):
    os.makedirs(save_path, exist_ok=True)
    ingestion.get_fbref_data(
        url=ingestion.fbref_url_builder(
            base_url=BASE_URL, league=LEAGUE_CONFIG, season=season
        ),
        league_name=LEAGUE_NAME,
        season=SEASON,
        dir=DATA_DIRECTORY,
    )


def process_historical_data():

    folder_path = Path("DATA/FBREF")
    files = [str(file) for file in folder_path.rglob("*.csv") if file.is_file()]

    candidates = []

    for file in files:
        if "unplayed" not in file:
            df = pd.read_csv(file, dtype={"Wk": int}).sort_values("Date")

            teams = set(df["Home"]).union(df["Away"])

            all_teams_stats = {team: stats.TeamStats(team, df) for team in teams}

            for fixture in df.values.tolist():

                week = fixture[0]
                date = fixture[2]
                home_team = fixture[4]
                away_team = fixture[5]
                fthg = fixture[6]
                ftag = fixture[7]

                home_rpi_df = stats.compute_rpi(
                    target_team_stats=all_teams_stats[home_team],
                    all_teams_stats=all_teams_stats,
                )

                away_rpi_df = stats.compute_rpi(
                    target_team_stats=all_teams_stats[away_team],
                    all_teams_stats=all_teams_stats,
                )

                # Shit RPi forward one to the value before the matches were played
                home_rpi_df["RPI"] = home_rpi_df["RPI"].shift(periods=1, fill_value=0)
                away_rpi_df["RPI"] = away_rpi_df["RPI"].shift(periods=1, fill_value=0)

                # "Team" is the opposition team in these dataframes
                match_home = home_rpi_df[
                    (home_rpi_df["Team"] == away_team) & (home_rpi_df["Date"] == date)
                ]
                match_away = away_rpi_df[
                    (away_rpi_df["Team"] == home_team) & (away_rpi_df["Date"] == date)
                ]

                home_rpi = match_home["RPI"].values[0]
                away_rpi = match_away["RPI"].values[0]

                rpi_diff = round(max(home_rpi, away_rpi) - min(home_rpi, away_rpi), 2)

                candidates.append(
                    {
                        "Wk": week,
                        "Date": date,
                        "League": file.split("/")[3].strip(".csv"),
                        "Home": home_team,
                        "Away": away_team,
                        "hRPI": home_rpi,
                        "aRPI": away_rpi,
                        "RPI_Diff": rpi_diff,
                        "FTHG": fthg,
                        "FTAG": ftag,
                        "FTR": "H" if fthg > ftag else "D" if fthg == ftag else "A",
                    }
                )

    # Short-list candidates to bet on
    candidates_df = pd.DataFrame(candidates)

    file_exists = os.path.exists("historical_candidates.csv")

    candidates_df.to_csv(
        "historical_candidates.csv", mode="a", header=not file_exists, index=False
    )

    candidates_df = (
        pd.read_csv("historical_candidates.csv")
        .sort_values("RPI_Diff")
        .to_csv("historical_candidates.csv", index=False)
    )


def analyse_historical_data():
    data = pd.read_csv("historical_candidates.csv")
    c1 = Counter(data["FTR"])
    p1 = sum(c1.values())
    c1p = [round((i / p1), 2) for i in list(c1.values())]
    print(c1p, c1p[1] + c1p[2])


def compute_rpi_and_generate_plots():

    historical_data_df = pd.read_csv(
        f"{DATA_DIRECTORY}/{HISTORICAL_DATA_FILE_NAME}", dtype={"Wk": int}
    )

    future_fixtures_df = pd.read_csv(
        f"{DATA_DIRECTORY}/{FUTURE_FIXTURES_FILE_NAME}", dtype={"Wk": int}
    )

    fixtures = future_fixtures_df[future_fixtures_df["Wk"].isin(WEEKS)]

    teams = set(historical_data_df["Home"]).union(historical_data_df["Away"])

    all_teams_stats = {
        team: stats.TeamStats(team, historical_data_df) for team in teams
    }

    candidates = []

    for fixture in fixtures.values.tolist():

        week = fixture[0]
        date = fixture[2]
        home_team = fixture[4]
        away_team = fixture[5]

        home_rpi_df = stats.compute_rpi(
            target_team_stats=all_teams_stats[home_team],
            all_teams_stats=all_teams_stats,
        )

        away_rpi_df = stats.compute_rpi(
            target_team_stats=all_teams_stats[away_team],
            all_teams_stats=all_teams_stats,
        )

        # Get the latest non-NaN RPI values for each team
        home_rpi_latest = home_rpi_df["RPI"][home_rpi_df["RPI"].last_valid_index()]
        away_rpi_latest = away_rpi_df["RPI"][away_rpi_df["RPI"].last_valid_index()]

        rpi_latest_diff = round(
            max(home_rpi_latest, away_rpi_latest)
            - min(home_rpi_latest, away_rpi_latest),
            2,
        )

        candidates.append(
            {
                "Wk": week,
                "Date": date,
                "League": LEAGUE_NAME,
                "Home": home_team,
                "Away": away_team,
                "hRPI": home_rpi_latest,
                "aRPI": away_rpi_latest,
                "RPI_Diff": rpi_latest_diff,
            }
        )

        # RPI PLOTS
        plotting.plot_compare_team_rolling_stats(
            dataframes=[home_rpi_df, away_rpi_df],
            teams=[home_team, away_team],
            target_stat="RPI",
            window=WINDOW,
            show=False,
            save_path=RPI_PLOTS_SAVE_DIRECORY,
            filename=f"{home_team}_{away_team}.png",
        )

        # PPG PLOTS
        plotting.plot_compare_team_rolling_stats(
            dataframes=[home_rpi_df, away_rpi_df],
            teams=[home_team, away_team],
            target_stat="PPG",
            window=WINDOW,
            show=False,
            save_path=PPG_PLOTS_SAVE_DIRECORY,
            filename=f"{home_team}_{away_team}",
        )

    # Short-list candidates to bet on
    candidates_df = pd.DataFrame(candidates)
    candidates_df = candidates_df[candidates_df["RPI_Diff"] <= RPI_DIFF_THRESHOLD]

    file_exists = os.path.exists("candidates.csv")

    candidates_df.to_csv(
        "candidates.csv", mode="a", header=not file_exists, index=False
    )

    candidates_df = (
        pd.read_csv("candidates.csv")
        .sort_values("RPI_Diff")
        .to_csv("candidates.csv", index=False)
    )


if __name__ == "__main__":

    # get_data(save_path=DATA_DIRECTORY)
    # compute_rpi_and_generate_plots()
    process_historical_data()
    # analyse_historical_data()
