from typing import Dict, List

import pandas as pd

from ..shared.exceptions import InsufficientDataException
from .entities import Match
from .repositories import MatchRepository


class PPICalculationService:
    def __init__(self, match_repository: MatchRepository):
        self.match_repository = match_repository

    def calculate_ppi_for_fixtures(
        self, league: str, fixtures: List[Match]
    ) -> List[Dict]:
        historical_matches = self.match_repository.get_by_league(league)
        played_matches = [m for m in historical_matches if m.is_played]

        if len(played_matches) < 10:
            raise InsufficientDataException(f"Not enough historical data for {league}")

        # Convert to DataFrame for PPI calculation (reusing existing logic)
        matches_df = self._matches_to_dataframe(played_matches)
        fixtures_df = self._matches_to_dataframe(fixtures)

        return self._calculate_ppi_records(matches_df, fixtures_df, league)

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

    def _calculate_ppi_records(
        self, matches_df: pd.DataFrame, fixtures_df: pd.DataFrame, league: str
    ) -> List[Dict]:
        from stats import compute_points_performance_index, compute_ppg

        home_ppg, away_ppg, total_ppg = compute_ppg(matches_df)
        candidates = []

        for _, fixture in fixtures_df.iterrows():
            try:
                home_ppi_df = compute_points_performance_index(
                    fixture["Home"], matches_df, home_ppg, away_ppg, total_ppg
                )
                away_ppi_df = compute_points_performance_index(
                    fixture["Away"], matches_df, home_ppg, away_ppg, total_ppg
                )

                home_ppi = home_ppi_df.tail(1)["PPI"].values[0]
                away_ppi = away_ppi_df.tail(1)["PPI"].values[0]

                candidates.append(
                    {
                        "Wk": fixture["Wk"],
                        "Date": fixture["Date"],
                        "League": league,
                        "Home": fixture["Home"],
                        "Away": fixture["Away"],
                        "hPPI": home_ppi,
                        "aPPI": away_ppi,
                        "PPI_Diff": abs(home_ppi - away_ppi),
                    }
                )
            except Exception:
                continue

        return candidates
