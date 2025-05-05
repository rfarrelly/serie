import time

import pandas as pd
from config import AppConfig, Leagues
from curl_cffi import requests
from utils.datetime_helpers import format_date
from utils.url_helpers import (
    fbduk_extra_url_builder,
    fbduk_main_url_builder,
    fbref_url_builder,
)


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
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                },
                impersonate="safari_ios",
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"An error occurred: {req_err}")

        columns = [
            "Wk",
            "League",
            "Season",
            "Day",
            "Date",
            "Time",
            "Home",
            "Score",
            "Away",
        ]

        data_df = pd.read_html(response.content)[0]

        data_df = data_df[
            ~data_df["Notes"].isin(["Match Suspended", "Match Cancelled"])
        ]

        data_df["League"] = league_name
        data_df["Season"] = season

        unplayed_fixtures_df = (
            data_df[data_df["Score"].isna()][columns]
            .drop("Score", axis="columns")
            .dropna(how="any", axis="index")
        )

        played_fixtures_df = data_df.dropna(
            how="any", subset=["Wk", "Score"], axis="index"
        )[columns]

        played_fixtures_df[["FTHG", "FTAG"]] = played_fixtures_df["Score"].apply(
            lambda x: pd.Series(self._parse_score(x))
        )
        played_fixtures_df = played_fixtures_df.drop("Score", axis="columns")

        self.write_files(played_fixtures_df, dir_path, league_name, season)
        self.write_files(
            unplayed_fixtures_df, dir_path, league_name, season, prefix="unplayed_"
        )
        time.sleep(3)

    def get_fbduk_data(self, league: Leagues, season: str):
        league_name = league.fbref_name

        if league.is_extra:
            url = fbduk_extra_url_builder(self.config.fbduk_base_url_extra, league)
            columns = ["Date", "Time", "Season", "Home", "Away", "PSCH", "PSCD", "PSCA"]
            season_extra_format = season.replace("-", "/")
        else:
            url = fbduk_main_url_builder(
                self.config.fbduk_base_url_main, league, season
            )
            columns = ["Date", "Time", "HomeTeam", "AwayTeam", "PSCH", "PSCD", "PSCA"]

        dir_path = self.config.get_fbduk_league_dir(league_name)

        data_df = pd.read_csv(url, encoding="latin-1")[columns].rename(
            columns={"HomeTeam": "Home", "AwayTeam": "Away"}
        )

        if "Season" in data_df.columns:
            data_df = data_df[data_df["Season"] == season_extra_format]

        data_df = format_date(data_df)

        self.write_files(data_df, dir_path, league_name, season)

    @staticmethod
    def _parse_score(score: str) -> tuple:
        if score:
            goals = score.split("–")
            return int(goals[0]), int(goals[1])
        return 0, 0
