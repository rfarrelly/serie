from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from domain.entities.match import Match, MatchResult
from domain.repositories import MatchRepository
from shared.types.common_types import LeagueName, Season, TeamName


class CSVMatchRepository(MatchRepository):
    """CSV implementation of match repository"""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    def get_matches_by_league_and_season(
        self, league: LeagueName, season: Season
    ) -> List[Match]:
        """Convert your existing CSV loading logic into domain objects"""

        # Use your existing file path logic
        file_path = self._data_dir / "FBREF" / league / f"{league}_{season}.csv"

        if not file_path.exists():
            return []

        # Load CSV (your existing pandas logic)
        df = pd.read_csv(file_path, dtype={"Wk": int})

        # Convert DataFrame rows to Match entities
        matches = []
        for _, row in df.iterrows():
            match = Match(
                home_team=TeamName(row["Home"]),
                away_team=TeamName(row["Away"]),
                league=league,
                date=pd.to_datetime(row["Date"]),
                week=int(row["Wk"]),
            )

            # Add result if match is completed
            if pd.notna(row["FTHG"]) and pd.notna(row["FTAG"]):
                match.record_result(
                    home_goals=int(row["FTHG"]), away_goals=int(row["FTAG"])
                )

            matches.append(match)

        return matches

    def get_matches_by_team(
        self, team: TeamName, league: LeagueName, season: Season
    ) -> List[Match]:
        all_matches = self.get_matches_by_league_and_season(league, season)
        return [
            match
            for match in all_matches
            if match.home_team == team or match.away_team == team
        ]
