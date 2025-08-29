from dataclasses import dataclass
from decimal import Decimal
from typing import List

from shared.types.common_types import BetType


@dataclass(frozen=True)
class Odds:
    """Immutable odds representation"""

    decimal: Decimal

    def __post_init__(self):
        if self.decimal <= 1.0:
            raise ValueError(f"Odds must be > 1.0, got {self.decimal}")

    @property
    def implied_probability(self) -> Decimal:
        """Convert odds to implied probability"""
        return Decimal("1.0") / self.decimal


@dataclass(frozen=True)
class MarketOdds:
    """Complete odds set for a match"""

    home: Odds
    draw: Odds
    away: Odds

    @property
    def overround(self) -> Decimal:
        """Calculate bookmaker's overround (profit margin)"""
        return (
            self.home.implied_probability
            + self.draw.implied_probability
            + self.away.implied_probability
        ) - Decimal("1.0")


@dataclass(frozen=True)
class Probability:
    """Probability value object with validation"""

    value: Decimal

    def __post_init__(self):
        if not (0 <= self.value <= 1):
            raise ValueError(f"Probability must be 0-1, got {self.value}")


@dataclass(frozen=True)
class MatchProbabilities:
    """Complete probability set for match outcomes"""

    home_win: Probability
    draw: Probability
    away_win: Probability

    def __post_init__(self):
        total = self.home_win.value + self.draw.value + self.away_win.value
        if abs(total - Decimal("1.0")) > Decimal("0.001"):
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")


@dataclass(frozen=True)
class Edge:
    """Betting edge calculation result"""

    value: Decimal
    bet_type: BetType

    def is_significant(self, threshold: Decimal = Decimal("0.02")) -> bool:
        """Domain rule: What constitutes a significant betting edge"""
        return self.value >= threshold
