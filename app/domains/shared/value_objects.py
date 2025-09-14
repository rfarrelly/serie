from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Odds:
    home: Decimal
    draw: Decimal
    away: Decimal

    def __post_init__(self):
        # Convert floats to Decimals if needed
        if isinstance(self.home, float):
            object.__setattr__(self, "home", Decimal(str(self.home)))
        if isinstance(self.draw, float):
            object.__setattr__(self, "draw", Decimal(str(self.draw)))
        if isinstance(self.away, float):
            object.__setattr__(self, "away", Decimal(str(self.away)))

        if any(float(odd) <= 1.0 for odd in [self.home, self.draw, self.away]):
            raise ValueError("All odds must be greater than 1.0")


@dataclass(frozen=True)
class Probabilities:
    home: Decimal
    draw: Decimal
    away: Decimal

    def __post_init__(self):
        total = self.home + self.draw + self.away
        if abs(total - 1.0) > 0.01:
            raise ValueError("Probabilities must sum to approximately 1.0")


@dataclass(frozen=True)
class TeamStats:
    goals_for: int
    goals_against: int
    points: int
    matches_played: int
