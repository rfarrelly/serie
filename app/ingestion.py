import time

import pandas as pd
from config import AppConfig, Leagues
from curl_cffi import requests
from utils.data_corrections import fix_scunthorpe_wealdestone_2025_2026
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
        print(f"File '{filename}' downloaded and saved to '{dir}'\r\n")

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
            "Home",
            "Score",
            "Away",
        ]

        data_df = pd.read_html(response.content)[0]

        data_df = data_df[
            ~data_df["Notes"].isin(["Match Suspended", "Match Cancelled"])
        ]

        if "Round" in data_df.columns:
            data_df = self._drop_non_regular_matches(data_df)

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

        played_fixtures_df = fix_scunthorpe_wealdestone_2025_2026(played_fixtures_df)

        self.write_files(played_fixtures_df, dir_path, league_name, season)
        self.write_files(
            unplayed_fixtures_df, dir_path, league_name, season, prefix="unplayed_"
        )
        time.sleep(3)

    def get_fbduk_data(self, league: Leagues, season: str):
        league_name = league.fbref_name
        odds_columns = [
            "PSH",
            "PSD",
            "PSA",
            "PSCH",
            "PSCD",
            "PSCA",
            "B365H",
            "B365D",
            "B365A",
            "B365CH",
            "B365CD",
            "B365CA",
        ]

        if league.is_extra:
            url = fbduk_extra_url_builder(self.config.fbduk_base_url_extra, league)
            columns = ["Date", "Season", "Home", "Away"] + odds_columns
            season_extra_format = season.replace("-", "/")
        else:
            url = fbduk_main_url_builder(
                self.config.fbduk_base_url_main, league, season
            )
            columns = ["Date", "HomeTeam", "AwayTeam"] + odds_columns

        dir_path = self.config.get_fbduk_league_dir(league_name)

        data_df = pd.read_csv(url, encoding="latin-1")[columns].rename(
            columns={"HomeTeam": "Home", "AwayTeam": "Away"}
        )

        # Fill empty closing odds with pre-closing odds and visa-versa
        data_df = self._fill_empty_odds(data_df)

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

    @staticmethod
    def _drop_non_regular_matches(df: pd.DataFrame):
        df = df.copy()
        # Identify the first value we want to keep
        first_value = df["Round"].iloc[0]

        # Keep rows until the first non-matching one
        return df[df["Round"] == first_value]

    @staticmethod
    def _fill_empty_odds(df: pd.DataFrame):
        df = df.copy()

        pinny_pairs = [("PSH", "PSCH"), ("PSD", "PSCD"), ("PSA", "PSCA")]
        b365_pairs = [("B365H", "B365CH"), ("B365D", "B365CD"), ("B365A", "B365CA")]

        for col1, col2 in pinny_pairs:
            # Fill col1 from col2 where col1 is NaN
            df[col1] = df[col1].fillna(df[col2])
            # Fill col2 from col1 where col2 is NaN
            df[col2] = df[col2].fillna(df[col1])

        for col1, col2 in b365_pairs:
            # Fill col1 from col2 where col1 is NaN
            df[col1] = df[col1].fillna(df[col2])
            # Fill col2 from col1 where col2 is NaN
            df[col2] = df[col2].fillna(df[col1])

        return df
