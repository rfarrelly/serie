from dataclasses import dataclass
from decimal import Decimal
from typing import Union


@dataclass(frozen=True)
class Odds:
    home: Decimal
    draw: Decimal
    away: Decimal

    def __post_init__(self):
        # Convert floats to Decimals if needed
        if isinstance(self.home, (float, int)):
            object.__setattr__(self, "home", Decimal(str(self.home)))
        if isinstance(self.draw, (float, int)):
            object.__setattr__(self, "draw", Decimal(str(self.draw)))
        if isinstance(self.away, (float, int)):
            object.__setattr__(self, "away", Decimal(str(self.away)))

        # Validate odds
        for odd in [self.home, self.draw, self.away]:
            if float(odd) <= 1.0:
                raise ValueError("All odds must be greater than 1.0")

    def to_probabilities(self) -> "Probabilities":
        """Convert odds to implied probabilities (with overround)."""
        home_prob = 1 / float(self.home)
        draw_prob = 1 / float(self.draw)
        away_prob = 1 / float(self.away)

        return Probabilities(
            home=Decimal(str(home_prob)),
            draw=Decimal(str(draw_prob)),
            away=Decimal(str(away_prob)),
        )


@dataclass(frozen=True)
class Probabilities:
    home: Decimal
    draw: Decimal
    away: Decimal

    def __post_init__(self):
        # Convert to Decimals if needed
        if isinstance(self.home, (float, int)):
            object.__setattr__(self, "home", Decimal(str(self.home)))
        if isinstance(self.draw, (float, int)):
            object.__setattr__(self, "draw", Decimal(str(self.draw)))
        if isinstance(self.away, (float, int)):
            object.__setattr__(self, "away", Decimal(str(self.away)))

        # Validate probabilities
        for prob in [self.home, self.draw, self.away]:
            if not (0 <= float(prob) <= 1):
                raise ValueError("All probabilities must be between 0 and 1")

        # Allow slight tolerance for rounding errors
        total = float(self.home + self.draw + self.away)
        if not (0.95 <= total <= 1.05):
            raise ValueError(
                f"Probabilities must sum to approximately 1.0 (got {total:.3f})"
            )

    def normalize(self) -> "Probabilities":
        """Return normalized probabilities that sum to exactly 1.0."""
        total = self.home + self.draw + self.away
        if total == 0:
            # Uniform distribution fallback
            return Probabilities(
                home=Decimal("0.333333"),
                draw=Decimal("0.333333"),
                away=Decimal("0.333334"),
            )

        return Probabilities(
            home=self.home / total, draw=self.draw / total, away=self.away / total
        )

    def to_odds(self) -> Odds:
        """Convert probabilities to fair odds."""
        # Use normalized probabilities
        normalized = self.normalize()

        return Odds(
            home=Decimal("1") / normalized.home,
            draw=Decimal("1") / normalized.draw,
            away=Decimal("1") / normalized.away,
        )


@dataclass(frozen=True)
class TeamStats:
    goals_for: int
    goals_against: int
    points: int
    matches_played: int

    def __post_init__(self):
        if self.matches_played < 0:
            raise ValueError("Matches played cannot be negative")
        if self.points < 0:
            raise ValueError("Points cannot be negative")
        if self.goals_for < 0 or self.goals_against < 0:
            raise ValueError("Goals cannot be negative")

    @property
    def points_per_game(self) -> float:
        """Calculate points per game."""
        return self.points / self.matches_played if self.matches_played > 0 else 0.0

    @property
    def goals_per_game(self) -> float:
        """Calculate goals for per game."""
        return self.goals_for / self.matches_played if self.matches_played > 0 else 0.0

    @property
    def goals_against_per_game(self) -> float:
        """Calculate goals against per game."""
        return (
            self.goals_against / self.matches_played if self.matches_played > 0 else 0.0
        )

    @property
    def goal_difference(self) -> int:
        """Calculate goal difference."""
        return self.goals_for - self.goals_against


@dataclass(frozen=True)
class BettingEdge:
    """Represents a betting edge calculation."""

    model_probability: Decimal
    market_probability: Decimal
    bookmaker_odds: Decimal
    edge_percentage: Decimal
    expected_value: Decimal

    def __post_init__(self):
        # Validate inputs
        for prob in [self.model_probability, self.market_probability]:
            if not (0 <= float(prob) <= 1):
                raise ValueError("Probabilities must be between 0 and 1")

        if float(self.bookmaker_odds) <= 1.0:
            raise ValueError("Bookmaker odds must be greater than 1.0")

    @property
    def kelly_fraction(self) -> Decimal:
        """Calculate Kelly criterion fraction for optimal bet sizing."""
        b = self.bookmaker_odds - Decimal("1")
        p = self.model_probability
        q = Decimal("1") - p

        if b <= 0:
            return Decimal("0")

        kelly = (b * p - q) / b
        return max(Decimal("0"), min(kelly, Decimal("0.25")))  # Cap at 25%

    @property
    def is_profitable(self) -> bool:
        """Check if this represents a profitable betting opportunity."""
        return float(self.edge_percentage) > 0 and float(self.expected_value) > 0
