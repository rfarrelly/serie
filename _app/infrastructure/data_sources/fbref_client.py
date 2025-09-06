import time
from pathlib import Path
from typing import Optional

import pandas as pd
from curl_cffi import requests


class FBRefClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def fetch_league_data(
        self, league_id: int, season: str, league_name: str
    ) -> Optional[pd.DataFrame]:
        url = f"{self.base_url}/{league_id}/{season}/schedule/{season}-{league_name}-Scores-and-Fixtures"

        try:
            response = requests.get(url, headers=self.headers, impersonate="safari_ios")
            response.raise_for_status()

            data_df = pd.read_html(response.content)[0]
            data_df = data_df[
                ~data_df.get("Notes", pd.Series()).isin(
                    ["Match Suspended", "Match Cancelled"]
                )
            ]

            if "Round" in data_df.columns:
                first_value = data_df["Round"].iloc[0]
                data_df = data_df[data_df["Round"] == first_value]

            return data_df

        except Exception as e:
            print(f"Error fetching FBRef data: {e}")
            return None

    def save_league_data(
        self, df: pd.DataFrame, output_dir: Path, league_name: str, season: str
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

        # Separate played and unplayed matches
        played_matches = df.dropna(subset=["Score"])
        unplayed_matches = df[df["Score"].isna()]

        # Process played matches
        if not played_matches.empty:
            played_matches[["FTHG", "FTAG"]] = played_matches["Score"].apply(
                lambda x: (
                    pd.Series(self._parse_score(x))
                    if pd.notna(x)
                    else pd.Series([None, None])
                )
            )
            played_matches = played_matches.drop("Score", axis=1)

            filename = output_dir / f"{league_name}_{season}.csv"
            played_matches.to_csv(filename, index=False)

        # Process unplayed matches
        if not unplayed_matches.empty:
            unplayed_matches = unplayed_matches.drop("Score", axis=1)
            filename = output_dir / f"unplayed_{league_name}_{season}.csv"
            unplayed_matches.to_csv(filename, index=False)

        time.sleep(3)  # Rate limiting

    @staticmethod
    def _parse_score(score: str) -> tuple:
        if score and "–" in score:
            goals = score.split("–")
            return int(goals[0]), int(goals[1])
        return 0, 0
