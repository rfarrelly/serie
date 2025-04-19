import pandas as pd
from curl_cffi import requests
from config import Leagues, AppConfig
from utils.url_helpers import fbref_url_builder


class DataIngestion:
    def __init__(self, config: AppConfig):
        self.config = config

    def write_files(
        self,
        df: pd.DataFrame,
        dir: str,
        league_name: str,
        season: str,
        prefix: str = None,
    ):
        filename = f"{league_name}_{season}.csv"
        if prefix:
            filename = prefix + filename
        df.to_csv(f"{dir}/{filename}", index=False)
        print(f"File '{filename}' downloaded and saved to '{dir}'")

    def get_fbref_data(self, league: Leagues, season: str):
        league_name = league.fbref_name
        url = fbref_url_builder(self.config.fbref_base_url, league, season)
        dir_path = self.config.get_fbref_league_dir(league_name)

        try:
            print(f"Getting data for url: {url}")
            response = requests.get(url, impersonate="safari_ios")
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"An error occurred: {req_err}")

        columns = ["Wk", "Day", "Date", "Time", "Home", "Score", "Away"]

        data_df = pd.read_html(response.content)[0]

        data_df = data_df[
            ~data_df["Notes"].isin(["Match Suspended", "Match Cancelled"])
        ]

        unplayed_fixtures_df = (
            data_df[data_df["Score"].isna()][columns]
            .drop("Score", axis="columns")
            .dropna(how="any", axis="index")
        )

        played_fixtures_df = data_df.dropna(how="any", subset="Score", axis="index")[
            columns
        ]

        played_fixtures_df[["FTHG", "FTAG"]] = played_fixtures_df["Score"].apply(
            lambda x: pd.Series(self._parse_score(x))
        )
        played_fixtures_df = played_fixtures_df.drop("Score", axis="columns")

        self.write_files(played_fixtures_df, dir_path, league_name, season)
        self.write_files(
            unplayed_fixtures_df, dir_path, league_name, season, prefix="unplayed_"
        )

    @staticmethod
    def _parse_score(score: str) -> tuple:
        if score:
            goals = score.split("–")
            return int(goals[0]), int(goals[1])
        return 0, 0
