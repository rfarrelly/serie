from pathlib import Path
from typing import Optional

import pandas as pd


class FBDukClient:
    def __init__(self, main_base_url: str, extra_base_url: str):
        self.main_base_url = main_base_url
        self.extra_base_url = extra_base_url

    def fetch_main_league_data(
        self, league_code: str, season: str
    ) -> Optional[pd.DataFrame]:
        season_year_max = season[2:4]
        season_year_min = season[7:9]
        url = (
            f"{self.main_base_url}/{season_year_max}{season_year_min}/{league_code}.csv"
        )

        try:
            df = pd.read_csv(url, encoding="latin-1")
            return self._process_odds_data(df)
        except Exception as e:
            print(f"Error fetching FBDuk main data: {e}")
            return None

    def fetch_extra_league_data(
        self, league_code: str, season: str
    ) -> Optional[pd.DataFrame]:
        url = f"{self.extra_base_url}/{league_code}.csv"

        try:
            df = pd.read_csv(url, encoding="latin-1")
            # Filter by season
            season_format = season.replace("-", "/")
            df = df[df["Season"] == season_format]
            return self._process_odds_data(df)
        except Exception as e:
            print(f"Error fetching FBDuk extra data: {e}")
            return None

    def _process_odds_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standardize column names
        df = df.rename(columns={"HomeTeam": "Home", "AwayTeam": "Away"})

        # Fill missing odds
        odds_pairs = [
            ("PSH", "PSCH"),
            ("PSD", "PSCD"),
            ("PSA", "PSCA"),
            ("B365H", "B365CH"),
            ("B365D", "B365CD"),
            ("B365A", "B365CA"),
        ]

        for col1, col2 in odds_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[col1] = df[col1].fillna(df[col2])
                df[col2] = df[col2].fillna(df[col1])

        # Format dates
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y").dt.strftime(
            "%Y-%m-%d"
        )

        return df

    def save_odds_data(
        self, df: pd.DataFrame, output_dir: Path, league_name: str, season: str
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{league_name}_{season}.csv"
        df.to_csv(filename, index=False)
