from app.common import ingestion, config, stats

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
    rpi = stats.compute_rpi(
        target_team_stats=all_teams_stats["Manchester Utd"],
        all_teams_stats=all_teams_stats,
    )
    print(all_teams_stats["Manchester Utd"].stats_df)
    print(rpi)
    # for team in teams:
    #     rpi = stats.compute_rpi(
    #         target_team_stats=all_teams_stats[team],
    #         all_teams_stats=all_teams_stats,
    #     )
    #     print(team, rpi.tail(1).values[0])


if __name__ == "__main__":
    main()
