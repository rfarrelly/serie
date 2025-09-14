from datetime import datetime
from typing import List

import pandas as pd
from domains.data.entities import Match, Team
from domains.data.repositories import MatchRepository

from ..adapters.file_adapter import CSVFileAdapter


class CSVMatchRepository(MatchRepository):
    def __init__(self, file_adapter: CSVFileAdapter, data_config):
        self.file_adapter = file_adapter
        self.data_config = data_config

    def get_by_date_range(self, start: datetime, end: datetime) -> List[Match]:
        # Load from latest PPI data
        if not self.file_adapter.file_exists("latest_ppi.csv"):
            return []

        df = self.file_adapter.read_csv("latest_ppi.csv")
        df["Date"] = pd.to_datetime(df["Date"])

        filtered = df[(df["Date"] >= start) & (df["Date"] <= end)]
        return self._dataframe_to_matches(filtered)

    def get_by_league(self, league: str) -> List[Match]:
        # Get all data files for the league
        matches = []

        # Try to load from multiple sources
        for file_pattern in ["**/**.csv"]:
            for file_path in self.data_config.fbref_data_dir.glob(file_pattern):
                if league.replace("-", "_") in str(file_path):
                    try:
                        df = pd.read_csv(file_path, dtype={"Wk": int})
                        league_matches = self._dataframe_to_matches(df, league)
                        matches.extend(league_matches)
                    except Exception:
                        continue

        return matches

    def get_historical_matches(self, league: str, seasons: List[str]) -> List[Match]:
        if self.file_adapter.file_exists("historical_ppi_and_odds.csv"):
            df = self.file_adapter.read_csv("historical_ppi_and_odds.csv")
            league_data = df[df["League"] == league]
            return self._dataframe_to_matches(league_data, league)
        return []

    def save_matches(self, matches: List[Match]) -> None:
        df = self._matches_to_dataframe(matches)
        self.file_adapter.write_csv(df, "latest_matches.csv")

    def _dataframe_to_matches(
        self, df: pd.DataFrame, league: str = None
    ) -> List[Match]:
        matches = []
        for _, row in df.iterrows():
            try:
                match_league = league or row.get("League", "Unknown")
                home_team = Team(name=row["Home"], league=match_league)
                away_team = Team(name=row["Away"], league=match_league)

                match = Match(
                    id=f"{row['Home']}_{row['Away']}_{row['Date']}",
                    date=pd.to_datetime(row["Date"]),
                    home_team=home_team,
                    away_team=away_team,
                    home_goals=row.get("FTHG"),
                    away_goals=row.get("FTAG"),
                    week=row.get("Wk"),
                )
                matches.append(match)
            except Exception:
                continue
        return matches

    def _matches_to_dataframe(self, matches: List[Match]) -> pd.DataFrame:
        data = []
        for match in matches:
            data.append(
                {
                    "Date": match.date,
                    "Home": match.home_team.name,
                    "Away": match.away_team.name,
                    "FTHG": match.home_goals,
                    "FTAG": match.away_goals,
                    "Wk": match.week,
                    "League": match.home_team.league,
                }
            )
        return pd.DataFrame(data)
