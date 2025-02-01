from app.common import ingestion, config, stats, plotting
import os
import pandas as pd

BASE_URL = "https://fbref.com/en/comps"
LEAGUE_CONFIG = config.League.EPL
LEAGUE_NAME = config.League.EPL.fbref_name
SEASON = "2024-2025"
DATA_STORAGE_DIRECTORY = f"./DATA/FBREF/{LEAGUE_NAME}"
DATA_FILE_NAME = f"{LEAGUE_NAME}_{SEASON}.csv"
RPI_PLOTS_SAVE_DIRECORY = f"./PLOTS/{LEAGUE_NAME}_{SEASON}/rpi"
PPG_PLOTS_SAVE_DIRECORY = f"./PLOTS/{LEAGUE_NAME}_{SEASON}/ppg"
TARGET_TEAMS = ["Wolves", "Aston Villa"]


def get_data(save_path: str):
    os.makedirs(save_path, exist_ok=True)
    ingestion.get_fbref_data(
        url=ingestion.fbref_url_builder(
            base_url=BASE_URL, league=LEAGUE_CONFIG, season=SEASON
        ),
        league_name=LEAGUE_NAME,
        season=SEASON,
        dir=DATA_STORAGE_DIRECTORY,
    )


def main():
    get_data(save_path=DATA_STORAGE_DIRECTORY)

    data = pd.read_csv(f"{DATA_STORAGE_DIRECTORY}/{DATA_FILE_NAME}", dtype={"Wk": int})
    teams = set(data["Home"]).union(data["Away"])

    all_teams_stats = {team: stats.TeamStats(team, data) for team in teams}
    dataframes = [
        stats.compute_rpi(
            target_team_stats=all_teams_stats[team],
            all_teams_stats=all_teams_stats,
        )
        for team in TARGET_TEAMS
    ]
    # RPI
    plotting.plot_compare_team_rolling_stats(
        dataframes=dataframes,
        teams=TARGET_TEAMS,
        target_stat="RPI",
        window=3,
        show=False,
        save_path=RPI_PLOTS_SAVE_DIRECORY,
        filename=f"{TARGET_TEAMS[0]}_{TARGET_TEAMS[1]}.png",
    )

    # PPG
    plotting.plot_compare_team_rolling_stats(
        dataframes=dataframes,
        teams=TARGET_TEAMS,
        target_stat="PPG",
        window=3,
        show=False,
        save_path=PPG_PLOTS_SAVE_DIRECORY,
        filename=f"{TARGET_TEAMS[0]}_{TARGET_TEAMS[1]}",
    )


if __name__ == "__main__":
    main()
