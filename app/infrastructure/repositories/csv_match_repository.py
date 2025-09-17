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
        """Load matches from fixtures with proper date filtering."""
        # Try to load from fixtures with PPI and odds first
        if self.file_adapter.file_exists("fixtures_ppi_and_odds.csv"):
            df = self.file_adapter.read_csv("fixtures_ppi_and_odds.csv")
        elif self.file_adapter.file_exists("latest_ppi.csv"):
            df = self.file_adapter.read_csv("latest_ppi.csv")
        else:
            print("No fixture files found")
            return []

        print(f"Loaded {len(df)} fixtures from file")

        # Convert dates and handle different date formats
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Filter out invalid dates
        valid_dates = df["Date"].notna()
        if not valid_dates.all():
            print(f"Warning: {(~valid_dates).sum()} fixtures have invalid dates")
            df = df[valid_dates]

        # Convert start and end to the same timezone-naive format
        start_date = (
            pd.to_datetime(start).date() if isinstance(start, str) else start.date()
        )
        end_date = pd.to_datetime(end).date() if isinstance(end, str) else end.date()

        # Filter by date range
        fixture_dates = df["Date"].dt.date
        date_mask = (fixture_dates >= start_date) & (fixture_dates <= end_date)
        filtered = df[date_mask]

        print(f"Date filtering: {start_date} to {end_date}")
        print(f"Available dates: {fixture_dates.min()} to {fixture_dates.max()}")
        print(f"After date filtering: {len(filtered)} fixtures")

        if len(filtered) == 0:
            print("No fixtures found in date range - checking all fixture dates:")
            for _, row in df.iterrows():
                print(f"  {row['Home']} vs {row['Away']}: {row['Date'].date()}")

        return self._dataframe_to_matches(filtered)

    def get_by_league(self, league: str) -> List[Match]:
        """Get all data files for the league."""
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

                # Handle date conversion
                match_date = row["Date"]
                if isinstance(match_date, str):
                    match_date = pd.to_datetime(match_date)
                elif pd.isna(match_date):
                    continue  # Skip matches with invalid dates

                match = Match(
                    id=f"{row['Home']}_{row['Away']}_{match_date.date()}",
                    date=match_date,
                    home_team=home_team,
                    away_team=away_team,
                    home_goals=row.get("FTHG"),
                    away_goals=row.get("FTAG"),
                    week=row.get("Wk"),
                )
                matches.append(match)
            except Exception as e:
                print(f"Error converting row to match: {e}")
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
