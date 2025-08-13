import numpy as np
import pandas as pd
from config import END_DATE, TODAY, AppConfig, Leagues
from ingestion import DataIngestion
from stats import compute_points_performance_index, compute_ppg
from utils.datetime_helpers import filter_date_range

# from zsd_poisson_model import RegularizedZSDPoissonModel


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

    def get_points_performance_index(self) -> dict:
        fixtures = filter_date_range(self.unplayed_matches_df, TODAY, END_DATE)

        if fixtures.empty:
            print("No Fixtures for this date range")
            return None

        home_ppg, away_ppg, total_ppg = compute_ppg(self.played_matches_df)

        candidates = []

        for fixture in fixtures.itertuples(index=False):
            week, date, home_team, away_team = (
                fixture.Wk,
                fixture.Date,
                fixture.Home,
                fixture.Away,
            )

            try:
                home_ppi_df = compute_points_performance_index(
                    home_team, self.played_matches_df, home_ppg, away_ppg, total_ppg
                )

                away_ppi_df = compute_points_performance_index(
                    away_team, self.played_matches_df, home_ppg, away_ppg, total_ppg
                )
            except:
                print(f"Error computing PPI for {self.league_name} - {date}")
                print(f"Continuing ...")
                continue

            home_ppi_latest = home_ppi_df.tail(1)["PPI"].values[0]
            away_ppi_latest = away_ppi_df.tail(1)["PPI"].values[0]

            ppi_diff = round(abs(home_ppi_latest - away_ppi_latest), 2)

            candidates.append(
                {
                    "Wk": week,
                    "Date": date,
                    "League": self.league_name,
                    "Home": home_team,
                    "Away": away_team,
                    "aOppPPG": away_ppi_df.tail(1)["OppPPG"].values[0],
                    "hOppPPG": home_ppi_df.tail(1)["OppPPG"].values[0],
                    "aPPG": away_ppi_df.tail(1)["PPG"].values[0],
                    "hPPG": home_ppi_df.tail(1)["PPG"].values[0],
                    "hPPI": home_ppi_latest,
                    "aPPI": away_ppi_latest,
                    "PPI_Diff": ppi_diff,
                }
            )

        candidates_df = pd.DataFrame(candidates)
        return candidates_df.to_dict(orient="records")

    # def get_zsd_poisson(self):
    #     fixtures = filter_date_range(self.unplayed_matches_df, TODAY, END_DATE)

    #     model = RegularizedZSDPoissonModel(played_matches=self.played_matches_df)

    #     results = []

    #     for fixture in fixtures.itertuples(index=False):
    #         week, date, home_team, away_team = (
    #             fixture.Wk,
    #             fixture.Date,
    #             fixture.Home,
    #             fixture.Away,
    #         )

    #         result = {}
    #         # Add fixture metadata
    #         result["Wk"] = week
    #         result["Date"] = date
    #         result["Home"] = home_team
    #         result["Away"] = away_team

    #         # Core predictions from the model
    #         pred_result = model.predict_match(
    #             home_team=home_team, away_team=away_team, max_goals=15
    #         )

    #         result |= pred_result

    #         results.append(result)

    #     preds_df = pd.DataFrame(results)
    #     return preds_df.to_dict(orient="records")


def get_historical_ppi(config: AppConfig) -> pd.DataFrame:
    print("Processing historical PPI")
    files = [
        str(file)
        for file in config.fbref_data_dir.rglob("*.csv")
        if file.is_file()
        if "unplayed" not in str(file)
    ]
    candidates = []

    for file in files:
        fixtures = pd.read_csv(file, dtype={"Wk": int}).sort_values("Date")

        teams = set(fixtures["Home"]).union(fixtures["Away"])

        hppg, appg, tppg = compute_ppg(fixtures)

        try:
            ppi_df_list = [
                compute_points_performance_index(
                    team, fixtures, hppg, appg, tppg
                ).sort_values("Date")
                for team in teams
            ]
        except:
            print(f"Error computing historical PPI for {file}")
            print(f"Continuing ...")
            continue

        for df in ppi_df_list:
            df[["OppPPG", "PPG", "PPI"]] = df[["OppPPG", "PPG", "PPI"]].shift(
                periods=1, fill_value=0
            )

        ppi_df = pd.concat(ppi_df_list)

        pivot_cols = ["OppPPG", "PPG", "PPI"]
        ppi_df_wide = ppi_df.pivot_table(
            index=[
                "Wk",
                "League",
                "Season",
                "Day",
                "Date",
                "Home",
                "Away",
                "FTHG",
                "FTAG",
            ],
            columns="TeamType",
            values=pivot_cols,
        )

        ppi_df_wide.columns = [f"{side}{col}" for col, side in ppi_df_wide.columns]
        ppi_final = ppi_df_wide.reset_index()

        ppi_final["PPI_Diff"] = round(abs(ppi_final["hPPI"] - ppi_final["aPPI"]), 2)

        candidates.append(ppi_final)

    candidates_df = pd.concat(candidates)
    print(f"Historical processor processed: {candidates_df.shape[0]} records")
    return candidates_df.sort_values("Date")
