# infrastructure/data/repositories/csv_match_repository.py
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from domain.entities.match import Fixture, Match
from domain.repositories import FixtureRepository, MatchRepository
from shared.types.common_types import LeagueName, Season, TeamName


class CSVMatchRepository(MatchRepository):
    """CSV implementation of match repository using your existing data structure"""

    def __init__(self, historical_data_path: str = "historical_ppi_and_odds.csv"):
        self._historical_data_path = historical_data_path
        self._data_cache: Optional[pd.DataFrame] = None

    def _load_data(self) -> pd.DataFrame:
        """Lazy load and cache the historical data"""
        if self._data_cache is None:
            if not Path(self._historical_data_path).exists():
                raise FileNotFoundError(
                    f"Historical data file not found: {self._historical_data_path}"
                )

            self._data_cache = pd.read_csv(
                self._historical_data_path, dtype={"Wk": int}
            )
            self._data_cache["Date"] = pd.to_datetime(self._data_cache["Date"])

        return self._data_cache

    def get_matches_by_league_and_season(
        self, league: LeagueName, season: Season
    ) -> List[Match]:
        """Load matches for a specific league and season"""
        df = self._load_data()

        # Filter by league and season
        filtered_df = df[
            (df["League"] == league) & (df["Season"] == season)
        ].sort_values("Date")

        # Convert DataFrame rows to Match entities
        matches = []
        for _, row in filtered_df.iterrows():
            match = Match.from_historical_csv_row(row)
            matches.append(match)

        return matches

    def get_matches_by_team(
        self, team: TeamName, league: LeagueName, season: Season
    ) -> List[Match]:
        """Get all matches for a specific team"""
        all_matches = self.get_matches_by_league_and_season(league, season)
        return [
            match
            for match in all_matches
            if match.home_team == team or match.away_team == team
        ]

    def get_recent_matches(self, team: TeamName, num_matches: int = 10) -> List[Match]:
        """Get the most recent matches for a team across all leagues/seasons"""
        df = self._load_data()

        # Filter by team and get most recent
        team_matches = (
            df[(df["Home"] == team) | (df["Away"] == team)]
            .sort_values("Date", ascending=False)
            .head(num_matches)
        )

        matches = []
        for _, row in team_matches.iterrows():
            match = Match.from_historical_csv_row(row)
            matches.append(match)

        return matches


class CSVFixtureRepository(FixtureRepository):
    """More robust CSV implementation with better error handling"""

    def __init__(
        self,
        fixtures_path: str = "fixtures_ppi_and_odds.csv",
        enhanced_path: str = "latest_zsd_enhanced.csv",
        candidates_path: str = "zsd_betting_candidates.csv",
    ):
        self._fixtures_path = fixtures_path
        self._enhanced_path = enhanced_path
        self._candidates_path = candidates_path

    def _safe_load_csv(
        self, filepath: str, required_cols: List[str]
    ) -> Optional[pd.DataFrame]:
        """Safely load CSV with validation"""
        if not Path(filepath).exists():
            print(f"⚠️  File not found: {filepath}")
            return None

        try:
            df = pd.read_csv(
                filepath, dtype={"Wk": int} if "Wk" in required_cols else {}
            )

            # Check for required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️  Missing columns in {filepath}: {missing_cols}")
                return None

            # Try to convert Date column safely
            if "Date" in df.columns:
                try:
                    df["Date"] = pd.to_datetime(df["Date"])
                except Exception as e:
                    print(f"⚠️  Date conversion error in {filepath}: {e}")
                    # Try alternative date formats
                    try:
                        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
                    except:
                        try:
                            df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
                        except Exception as final_e:
                            print(
                                f"❌ Could not convert dates in {filepath}: {final_e}"
                            )
                            return None

            return df

        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None

    def get_upcoming_fixtures(
        self,
        league: Optional[LeagueName] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> List[Fixture]:
        """Get upcoming fixtures with robust error handling"""

        required_cols = ["Date", "Home", "Away", "League"]
        df = self._safe_load_csv(self._fixtures_path, required_cols)

        if df is None:
            return []

        # Apply filters
        try:
            if league:
                df = df[df["League"] == league]

            if from_date:
                df = df[df["Date"] >= from_date]

            if to_date:
                df = df[df["Date"] <= to_date]

            # Sort by date
            df = df.sort_values("Date")

            # Convert to Fixture entities
            fixtures = []
            for _, row in df.iterrows():
                try:
                    fixture = Fixture.from_ppi_csv_row(row)
                    fixtures.append(fixture)
                except Exception as e:
                    print(f"⚠️  Skipping row due to error: {e}")
                    continue

            return fixtures

        except Exception as e:
            print(f"❌ Error processing fixtures: {e}")
            return []

    def get_fixtures_with_predictions(self) -> List[Fixture]:
        """Get fixtures that have model predictions"""

        required_cols = ["Date", "Home", "Away", "League"]
        df = self._safe_load_csv(self._enhanced_path, required_cols)

        if df is None:
            print("📝 No enhanced predictions available")
            return []

        fixtures = []
        for _, row in df.iterrows():
            try:
                fixture = Fixture.from_enhanced_csv_row(row)
                fixtures.append(fixture)
            except Exception as e:
                print(f"⚠️  Skipping prediction row: {e}")
                continue

        return fixtures

    def get_betting_candidates(self, min_edge: float = 0.02) -> List[Fixture]:
        """Get fixtures identified as betting candidates"""

        # First try candidates file
        required_cols = ["Date", "Home", "Away", "League"]
        df = self._safe_load_csv(self._candidates_path, required_cols)

        if df is not None:
            # Filter by edge if column exists
            if "Edge" in df.columns:
                df = df[df["Edge"] >= min_edge]

            fixtures = []
            for _, row in df.iterrows():
                try:
                    fixture = Fixture.from_enhanced_csv_row(row)
                    fixtures.append(fixture)
                except Exception as e:
                    print(f"⚠️  Skipping candidate row: {e}")
                    continue

            return fixtures

        # Fall back to enhanced data
        print("📝 Falling back to enhanced data for betting candidates")
        df = self._safe_load_csv(self._enhanced_path, required_cols)

        if df is None:
            return []

        # Filter by betting criteria
        if "Is_Betting_Candidate" in df.columns:
            df = df[df["Is_Betting_Candidate"] == True]
        elif "Edge" in df.columns:
            df = df[df["Edge"] >= min_edge]

        fixtures = []
        for _, row in df.iterrows():
            try:
                fixture = Fixture.from_enhanced_csv_row(row)
                fixtures.append(fixture)
            except Exception as e:
                print(f"⚠️  Skipping enhanced row: {e}")
                continue

        return fixtures
