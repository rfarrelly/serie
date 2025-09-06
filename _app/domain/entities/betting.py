from dataclasses import dataclass
from datetime import datetime

from ..value_objects.odds import Odds
from ..value_objects.probability import Probability


@dataclass
class BettingOpportunity:
    match_id: str
    bet_type: str
    stake: float
    odds: Odds
    model_probability: Probability
    market_probability: Probability
    edge: float
    expected_value: float

    @property
    def is_profitable(self) -> bool:
        return self.edge > 0


@dataclass
class BettingResult:
    opportunity: BettingOpportunity
    actual_outcome: str
    profit: float
    date_placed: datetime

    @property
    def won(self) -> bool:
        return self.profit > 0
