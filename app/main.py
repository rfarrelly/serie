import stats
import pandas as pd
import ingestion
from config import Leagues, AppConfig, DEFAULT_CONFIG, TIME_DELTA
from datetime import datetime, timedelta

TODAY = datetime.now().date()


def filter_date_range(df, date_column):
    end_date = TODAY + timedelta(days=TIME_DELTA)

    df[date_column] = pd.to_datetime(df[date_column])

    filtered_df = df[
        (df[date_column].dt.date >= TODAY) & (df[date_column].dt.date <= end_date)
    ]

    return filtered_df


class LeagueProcessor:
    def __init__(self, league: Leagues, config: AppConfig):
        self.league = league
        self.config = config
        self.league_name = league.fbref_name
        self.fbref_dir = config.get_fbref_league_dir(self.league_name)

    def get_data(self):
        ingestion.get_fbref_data(
            url=ingestion.fbref_url_builder(
                base_url=self.config.fbref_base_url,
                league=self.league,
                season=self.config.current_season,
            ),
            league_name=self.league_name,
            season=self.config.current_season,
            dir=self.fbref_dir,
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
        all_teams_stats = {team: stats.TeamStats(team, historical_df) for team in teams}

        candidates = []

        for fixture in fixtures.itertuples(index=False):
            week, date, time, home_team, away_team = (
                fixture.Wk,
                fixture.Date,
                fixture.Time,
                fixture.Home,
                fixture.Away,
            )
            home_rpi_latest = stats.compute_rpi(
                all_teams_stats[home_team], all_teams_stats
            )["RPI"].iloc[-1]
            away_rpi_latest = stats.compute_rpi(
                all_teams_stats[away_team], all_teams_stats
            )["RPI"].iloc[-1]
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


def main():
    all_candidates = []

    for league in Leagues:

        processor = LeagueProcessor(league, DEFAULT_CONFIG)

        print(f"Processing {league.name} ({league.value['fbref_name']})")

        processor.get_data()

        league_candidates = processor.compute_league_rpi()

        if league_candidates:
            all_candidates.extend(league_candidates)

    if all_candidates:
        candidates_df = pd.DataFrame(all_candidates).sort_values(by="RPI_Diff")
        candidates_df = candidates_df[
            candidates_df["RPI_Diff"] <= DEFAULT_CONFIG.rpi_diff_threshold
        ]
        candidates_df.to_csv("candidates.csv", index=False)
        print("Saved sorted candidate matches to candidates.csv")


if __name__ == "__main__":
    main()
