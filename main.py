from app.common import ingestion, config, stats, plotting

import pandas as pd

FBREF_LEAGUE_NAME = config.FbrefLeagueName.EPL
FBREF_DATA_DIRECTORY = "app/DATA/FBREF"
SEASON = "2024-2025"

data = pd.read_csv("app/DATA/FBREF/Premier_League_2024_2025.csv", dtype={"Wk": int})
teams = set(data["HomeTeam"]).union(data["AwayTeam"])


def main():
    # ingestion.get_fbref_data(
    #     url=config.fbref_url_builder(league=FBREF_LEAGUE_NAME, season=SEASON),
    #     league=FBREF_LEAGUE_NAME,
    #     season=SEASON,
    #     dir=FBREF_DATA_DIRECTORY,
    # )

    all_teams_stats = {team: stats.TeamStats(team, data) for team in teams}
    target_teams = ["Bournemouth", "Liverpool"]
    dataframes = [
        stats.compute_rpi(
            target_team_stats=all_teams_stats[team],
            all_teams_stats=all_teams_stats,
        )
        for team in target_teams
    ]
    plotting.plot_compare_team_rolling_stats(dataframes, target_teams, "RPI", 3)


if __name__ == "__main__":
    main()
