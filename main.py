from app.common import ingestion, config, stats

import pandas as pd

FBREF_LEAGUE_NAME = config.FbrefLeagueName.EPL
FBREF_DATA_DIRECTORY = "app/DATA/FBREF"
SEASON = "2024-2025"

data = pd.read_csv("app/DATA/FBREF/Premier_League_2024_2025.csv")
teams = set(data["HomeTeam"]).union(data["AwayTeam"])


def main():
    # ingestion.get_fbref_data(
    #     url=config.fbref_url_builder(league=FBREF_LEAGUE_NAME, season=SEASON),
    #     league=FBREF_LEAGUE_NAME,
    #     season=SEASON,
    #     dir=FBREF_DATA_DIRECTORY,
    # )

    team_stats = {team: stats.TeamStats(team, data) for team in teams}
    print(team_stats["Arsenal"].get_stats_df)


if __name__ == "__main__":
    main()
