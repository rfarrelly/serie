from app.common import ingestion, config, stats, plotting
import os
import pandas as pd

BASE_URL = "https://fbref.com/en/comps"
LEAGUE_CONFIG = config.League.EPL
LEAGUE_NAME = LEAGUE_CONFIG.fbref_name
SEASON = "2024-2025"
DATA_DIRECTORY = f"./DATA/FBREF/{LEAGUE_NAME}"
HISTORICAL_DATA_FILE_NAME = f"{LEAGUE_NAME}_{SEASON}.csv"
FUTURE_FIXTURES_FILE_NAME = f"unplayed_{LEAGUE_NAME}_{SEASON}.csv"
RPI_PLOTS_SAVE_DIRECORY = f"./PLOTS/{LEAGUE_NAME}_{SEASON}/rpi"
PPG_PLOTS_SAVE_DIRECORY = f"./PLOTS/{LEAGUE_NAME}_{SEASON}/ppg"
WEEKS = [15, 25]
WINDOW = 5


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


def main():
    # get_data(save_path=DATA_STORAGE_DIRECTORY)

    historical_data_df = pd.read_csv(
        f"{DATA_DIRECTORY}/{HISTORICAL_DATA_FILE_NAME}", dtype={"Wk": int}
    )

    future_fixtures_df = pd.read_csv(
        f"{DATA_DIRECTORY}/{FUTURE_FIXTURES_FILE_NAME}", dtype={"Wk": int}
    )

    target_teams = future_fixtures_df[future_fixtures_df["Wk"].isin(WEEKS)][
        ["Home", "Away"]
    ].values.tolist()

    teams = set(historical_data_df["Home"]).union(historical_data_df["Away"])

    all_teams_stats = {
        team: stats.TeamStats(team, historical_data_df) for team in teams
    }

    for fixture in target_teams:
        dataframes = []
        for team in fixture:
            dataframes.append(
                stats.compute_rpi(
                    target_team_stats=all_teams_stats[team],
                    all_teams_stats=all_teams_stats,
                )
            )

        # RPI
        plotting.plot_compare_team_rolling_stats(
            dataframes=dataframes,
            teams=fixture,
            target_stat="RPI",
            window=WINDOW,
            show=False,
            save_path=RPI_PLOTS_SAVE_DIRECORY,
            filename=f"{fixture[0]}_{fixture[1]}.png",
        )

        # PPG
        plotting.plot_compare_team_rolling_stats(
            dataframes=dataframes,
            teams=fixture,
            target_stat="PPG",
            window=WINDOW,
            show=False,
            save_path=PPG_PLOTS_SAVE_DIRECORY,
            filename=f"{fixture[0]}_{fixture[1]}",
        )


if __name__ == "__main__":
    main()
