"""
Compatibility layer for backtesting.py - provides the expected interface
while delegating to the domain services.
"""

from typing import Dict, Optional

import pandas as pd
from domains.betting.services import BettingAnalysisService as DomainBettingService


class BettingCalculator:
    """Compatibility wrapper for the domain BettingAnalysisService."""

    def __init__(self):
        self.bet_types = ["Home", "Draw", "Away"]
        self._domain_service = DomainBettingService()

    def analyze_prediction(self, prediction_row: pd.Series) -> Optional[Dict]:
        """Analyze a single prediction for betting opportunities."""
        return self._domain_service.analyze_prediction_row(prediction_row)

    def find_betting_candidates(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Find betting candidates from predictions dataframe."""
        return self._domain_service.find_betting_candidates(predictions_df)

    def _metrics_to_dict(self, metrics: Dict) -> Dict:
        """Convert betting metrics to dictionary for DataFrame construction."""
        return self._domain_service._metrics_to_dict(metrics)


class BettingFilter:
    """Filters betting candidates based on various criteria."""

    def __init__(
        self, min_edge: float = 0.02, min_prob: float = 0.1, max_odds: float = 10.0
    ):
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.max_odds = max_odds

    def filter_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to betting candidates."""
        if len(candidates_df) == 0:
            return candidates_df

        filtered = candidates_df[
            (candidates_df["Edge"] >= self.min_edge)
            & (candidates_df["Model_Prob"] >= self.min_prob)
            & (candidates_df["Soft_Odds"] <= self.max_odds)
        ].copy()

        return filtered.sort_values("Edge", ascending=False)
