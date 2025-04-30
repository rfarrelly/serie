from config import Leagues, AppConfig, TODAY, END_DATE
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

    @property
    def played_matches_df(self):
        return pd.read_csv(
            self.fbref_dir / f"{self.league_name}_{self.config.current_season}.csv",
            dtype={"Wk": int},
        )

    @property
    def unplayed_matches_df(self):
        return pd.read_csv(
            self.fbref_dir
            / f"unplayed_{self.league_name}_{self.config.current_season}.csv",
            dtype={"Wk": int},
        )

    def get_fbref_data(self):
        self.ingestion.get_fbref_data(
            league=self.league, season=self.config.current_season
        )

    def get_fbduk_data(self):
        self.ingestion.get_fbduk_data(
            league=self.league, season=self.config.current_season
        )

    def generate_bet_candidates(self) -> dict:

        fixtures = filter_date_range(self.unplayed_matches_df, TODAY, END_DATE)

        teams = set(self.played_matches_df["Home"]).union(
            self.played_matches_df["Away"]
        )
        all_teams_stats = {
            team: TeamStats(team, self.played_matches_df) for team in teams
        }

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


def process_historical_data(config: AppConfig) -> pd.DataFrame:
    print("Processing historical data")
    files = [
        str(file)
        for file in config.fbref_data_dir.rglob("*.csv")
        if file.is_file()
        if "unplayed" not in str(file)
    ]
    candidates = []
    input_count = 0

    for file in files:

        fixtures = pd.read_csv(file, dtype={"Wk": int}).sort_values("Date")

        input_count += fixtures.shape[0]

        teams = set(fixtures["Home"]).union(fixtures["Away"])

        all_teams_stats = {team: TeamStats(team, fixtures) for team in teams}

        rpi_df_dict = {
            team: compute_rpi(all_teams_stats[team], all_teams_stats) for team in teams
        }

        # Shift RPI forward 1 place for analysis
        for team, df in rpi_df_dict.items():
            df["RPI"] = df["RPI"].shift(periods=1, fill_value=0)

        for fixture in fixtures.itertuples(index=False):
            week, date, time, league, home_team, away_team, fthg, ftag = (
                fixture.Wk,
                fixture.Date,
                fixture.Time,
                fixture.League,
                fixture.Home,
                fixture.Away,
                fixture.FTHG,
                fixture.FTAG,
            )

            home_rpi_df = rpi_df_dict[home_team]
            away_rpi_df = rpi_df_dict[away_team]

            home_rpi_df = home_rpi_df[
                (home_rpi_df["Team"] == away_team) & (home_rpi_df["Date"] == date)
            ]
            away_rpi_df = away_rpi_df[
                (away_rpi_df["Team"] == home_team) & (away_rpi_df["Date"] == date)
            ]

            home_rpi = home_rpi_df["RPI"].values[0]
            away_rpi = away_rpi_df["RPI"].values[0]

            rpi_diff = round(abs(home_rpi - away_rpi), 2)

            candidates.append(
                {
                    "Wk": week,
                    "Date": date,
                    "Time": time,
                    "League": league,
                    "Home": home_team,
                    "Away": away_team,
                    "FTHG": fthg,
                    "FTAG": ftag,
                    "hRPI": float(home_rpi),
                    "aRPI": float(away_rpi),
                    "RPI_Diff": rpi_diff,
                    "FTR": "H" if fthg > ftag else "D" if fthg == ftag else "A",
                }
            )
    candidates_df = pd.DataFrame(candidates)
    print(f"Historical processor produced: {candidates_df.shape[0]} records")
    return candidates_df.sort_values("Date")
