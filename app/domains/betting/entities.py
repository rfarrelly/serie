from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..predictions.entities import Prediction
from ..shared.value_objects import Odds


@dataclass
class BettingOpportunity:
    prediction: Prediction
    recommended_outcome: str  # 'H', 'D', 'A'
    edge: float
    fair_odds: float
    market_odds: Odds
    expected_value: float
    kelly_fraction: Optional[float] = None


@dataclass
class BettingResult:
    opportunity: BettingOpportunity
    stake: float
    outcome: str
    profit: float
    placed_at: datetime
