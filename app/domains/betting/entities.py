from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from ..predictions.entities import Prediction
from ..shared.value_objects import Odds


@dataclass
class BettingOpportunity:
    prediction: Prediction
    recommended_outcome: str  # 'H', 'D', 'A'
    edge: Decimal
    fair_odds: Decimal
    market_odds: Odds
    expected_value: Decimal
    kelly_fraction: Optional[Decimal] = None


@dataclass
class BettingResult:
    opportunity: BettingOpportunity
    stake: Decimal
    outcome: str
    profit: Decimal
    placed_at: datetime
