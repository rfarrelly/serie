from config import Leagues, AppConfig
from ingestion import DataIngestion
from stats import TeamStats, compute_rpi
from utils.datetime_helpers import filter_date_range
import pandas as pd


class LeagueProcessor:
    def __init__(self, league: Leagues, config: AppConfig):
        self.league = league
        self.config = config
        self.league_name = league.fbref_name
        self.fbref_dir = config.get_fbref_league_dir(self.league_name)
        self.ingestion = DataIngestion(config)

    def get_data(self):
        self.ingestion.get_fbref_data(
            league=self.league, season=self.config.current_season
        )

    def compute_league_rpi(self):
        played_fixtures_file = (
            self.fbref_dir / f"{self.league_name}_{self.config.current_season}.csv"
        )
        future_fixtures_file = (
            self.fbref_dir
            / f"unplayed_{self.league_name}_{self.config.current_season}.csv"
        )

        historical_df = pd.read_csv(played_fixtures_file, dtype={"Wk": int})
        future_fixtures_df = pd.read_csv(future_fixtures_file, dtype={"Wk": int})
        fixtures = filter_date_range(future_fixtures_df, "Date")

        teams = set(historical_df["Home"]).union(historical_df["Away"])
        all_teams_stats = {team: TeamStats(team, historical_df) for team in teams}

        candidates = []

        for fixture in fixtures.itertuples(index=False):
            week, date, time, home_team, away_team = (
                fixture.Wk,
                fixture.Date,
                fixture.Time,
                fixture.Home,
                fixture.Away,
            )
            home_rpi_latest = compute_rpi(all_teams_stats[home_team], all_teams_stats)[
                "RPI"
            ].iloc[-1]
            away_rpi_latest = compute_rpi(all_teams_stats[away_team], all_teams_stats)[
                "RPI"
            ].iloc[-1]
            rpi_diff = round(abs(home_rpi_latest - away_rpi_latest), 2)

            candidates.append(
                {
                    "Wk": week,
                    "Date": date,
                    "Time": time,
                    "League": self.league_name,
                    "Home": home_team,
                    "Away": away_team,
                    "hRPI": home_rpi_latest,
                    "aRPI": away_rpi_latest,
                    "RPI_Diff": rpi_diff,
                }
            )

        candidates_df = pd.DataFrame(candidates)
        return candidates_df.to_dict(orient="records")
