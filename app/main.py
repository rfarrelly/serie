import stats
import pandas as pd
import ingestion
from config import Leagues, AppConfig, LEAGUE_WEEKS, DEFAULT_CONFIG


class LeagueProcessor:
    def __init__(self, league: Leagues, config: AppConfig):
        self.league = league
        self.config = config
        self.league_name = league.fbref_name
        self.fbref_dir = config.get_fbref_league_dir(self.league_name)
        self.fbduk_dir = config.get_fbduk_league_dir(self.league_name)

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

        ingestion.get_fbduk_data(
            url=ingestion.fbduk_url_builder(
                base_url=self.config.fbduk_base_url,
                league=self.league,
                season=self.config.current_season,
            ),
            league_name=self.league_name,
            season=self.config.current_season,
            dir=self.fbduk_dir,
        )

    def compute_league_rpi(self, weeks):
        data_file = (
            self.fbref_dir / f"{self.league_name}_{self.config.current_season}.csv"
        )
        future_fixtures_file = (
            self.fbref_dir
            / f"unplayed_{self.league_name}_{self.config.current_season}.csv"
        )

        historical_df = pd.read_csv(data_file, dtype={"Wk": int})
        future_fixtures_df = pd.read_csv(future_fixtures_file, dtype={"Wk": int})
        fixtures = future_fixtures_df[future_fixtures_df["Wk"].isin(weeks)]
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


def process_historical_data(config: AppConfig):
    folder_path = config.fbref_data_dir
    files = [str(file) for file in folder_path.rglob("*.csv") if file.is_file()]

    candidates = []

    for file in files:
        if "unplayed" not in file:
            print(f"Processing {file}")
            df = pd.read_csv(file, dtype={"Wk": int}).sort_values("Date")

            teams = set(df["Home"]).union(df["Away"])

            all_teams_stats = {team: stats.TeamStats(team, df) for team in teams}

            rpi_df_dict = {}

            for fixture in df.itertuples(index=False):
                week, date, home_team, away_team, fthg, ftag = (
                    fixture.Wk,
                    fixture.Date,
                    fixture.Home,
                    fixture.Away,
                    fixture.FTHG,
                    fixture.FTAG,
                )

                if home_team in rpi_df_dict.keys():
                    home_rpi_df = rpi_df_dict[home_team]
                else:
                    home_rpi_df = stats.compute_rpi(
                        target_team_stats=all_teams_stats[home_team],
                        all_teams_stats=all_teams_stats,
                    )
                    home_rpi_df["RPI"] = home_rpi_df["RPI"].shift(
                        periods=1, fill_value=0
                    )
                    rpi_df_dict[home_team] = home_rpi_df

                if away_team in rpi_df_dict.keys():
                    away_rpi_df = rpi_df_dict[away_team]
                else:
                    away_rpi_df = stats.compute_rpi(
                        target_team_stats=all_teams_stats[away_team],
                        all_teams_stats=all_teams_stats,
                    )
                    away_rpi_df["RPI"] = away_rpi_df["RPI"].shift(
                        periods=1, fill_value=0
                    )
                    rpi_df_dict[away_team] = away_rpi_df

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
    return pd.DataFrame(candidates)


def main():
    all_candidates = []

    for league in Leagues:
        league_weeks = LEAGUE_WEEKS.get(league)
        processor = LeagueProcessor(league, DEFAULT_CONFIG)

        print(f"Processing {league.name} ({league.value['fbref_name']})")

        # Run this once per league
        processor.get_data()

        # Now compute RPI differences for upcoming fixtures
        if league_weeks:
            league_candidates = processor.compute_league_rpi(league_weeks)
            if league_candidates:
                all_candidates.extend(league_candidates)

    if all_candidates:
        candidates_df = pd.DataFrame(all_candidates).sort_values(by="RPI_Diff")
        candidates_df = candidates_df[
            candidates_df["RPI_Diff"] <= DEFAULT_CONFIG.rpi_diff_threshold
        ]
        candidates_df.to_csv("candidates.csv", index=False)
        print("Saved sorted candidate matches to candidates.csv")

    process_historical_data(DEFAULT_CONFIG).to_csv("historical.csv", index=False)
    print("Saved sorted historical matches to historical.csv")


if __name__ == "__main__":
    main()
