import stats
import os
import pandas as pd

import config, ingestion, plotting

BASE_URL = "https://fbref.com/en/comps"
LEAGUE_CONFIG = config.League.ECH
LEAGUE_NAME = LEAGUE_CONFIG.fbref_name
SEASON = "2024-2025"
DATA_DIRECTORY = f"./DATA/FBREF/{LEAGUE_NAME}"
HISTORICAL_DATA_FILE_NAME = f"{LEAGUE_NAME}_{SEASON}.csv"
FUTURE_FIXTURES_FILE_NAME = f"unplayed_{LEAGUE_NAME}_{SEASON}.csv"
RPI_PLOTS_SAVE_DIRECORY = f"./PLOTS/{LEAGUE_NAME}_{SEASON}/rpi"
PPG_PLOTS_SAVE_DIRECORY = f"./PLOTS/{LEAGUE_NAME}_{SEASON}/ppg"
WEEKS = [38]
WINDOW = 1


def get_data(save_path: str):
    os.makedirs(save_path, exist_ok=True)
    ingestion.get_fbref_data(
        url=ingestion.fbref_url_builder(
            base_url=BASE_URL, league=LEAGUE_CONFIG, season=SEASON
        ),
        league_name=LEAGUE_NAME,
        season=SEASON,
        dir=DATA_DIRECTORY,
    )


def process_historical_data():

    files = [
        os.path.join(DATA_DIRECTORY, f)
        for f in os.listdir(DATA_DIRECTORY)
        if os.path.isfile(os.path.join(DATA_DIRECTORY, f))
    ]

    for file in files:
        if "unplayed" not in file:
            df = pd.read_csv(file, dtype={"Wk": int}).sort_values("Date")

            teams = set(df["Home"]).union(df["Away"])

            fixtures = df[["Wk", "Date", "Home", "Away"]].values.tolist()

            all_teams_stats = {team: stats.TeamStats(team, df) for team in teams}

            for fixture in fixtures:
                dataframes = []
                for team in fixture[2:]:
                    dataframes.append(
                        stats.compute_rpi(
                            target_team_stats=all_teams_stats[team],
                            all_teams_stats=all_teams_stats,
                        )
                    )

                rpi_df = pd.merge(
                    dataframes[0].rename(columns={"RPI": "hRPI"}),
                    dataframes[1].rename(columns={"RPI": "aRPI"}),
                    on="Wk",
                    how="outer",
                )

                rpi_df = rpi_df.drop(columns=rpi_df.filter(regex=r"^\d+_[xy]$").columns)

                rpi_df[["hRPI", "aRPI"]] = rpi_df[["hRPI", "aRPI"]].ffill()

                breakpoint()


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

    for fixture in fixtures.values.tolist():
        home_rpi_df = stats.compute_rpi(
            target_team_stats=all_teams_stats[fixture[4]],
            all_teams_stats=all_teams_stats,
        )

        away_rpi_df = stats.compute_rpi(
            target_team_stats=all_teams_stats[fixture[5]],
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

        file_exists = os.path.exists("candidates.csv")

        data = {
            "Wk": [fixture[0]],
            "Date": [fixture[2]],
            "League": [LEAGUE_NAME],
            "Home": [fixture[4]],
            "Away": [fixture[5]],
            "hRPI": [home_rpi_latest],
            "aRPI": [away_rpi_latest],
            "RPI_Diff": [rpi_latest_diff],
        }

        df = pd.DataFrame(data)
        df = df[df["RPI_Diff"] <= 0.1]

        df.to_csv("candidates.csv", mode="a", header=not file_exists, index=False)

        df = (
            pd.read_csv("candidates.csv")
            .sort_values("RPI_Diff")
            .to_csv("candidates.csv", index=False)
        )

        # RPI
        plotting.plot_compare_team_rolling_stats(
            dataframes=[home_rpi_df, away_rpi_df],
            teams=[fixture[4], fixture[5]],
            target_stat="RPI",
            window=WINDOW,
            show=False,
            save_path=RPI_PLOTS_SAVE_DIRECORY,
            filename=f"{fixture[0]}_{fixture[1]}.png",
        )

        # PPG
        plotting.plot_compare_team_rolling_stats(
            dataframes=[home_rpi_df, away_rpi_df],
            teams=[fixture[4], fixture[5]],
            target_stat="PPG",
            window=WINDOW,
            show=False,
            save_path=PPG_PLOTS_SAVE_DIRECORY,
            filename=f"{fixture[0]}_{fixture[1]}",
        )


if __name__ == "__main__":

    # get_data(save_path=DATA_DIRECTORY)

    compute_rpi_and_generate_plots()
    # process_historical_data()
