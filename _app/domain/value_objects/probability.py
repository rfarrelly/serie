from dataclasses import dataclass


@dataclass(frozen=True)
class Probability:
    value: float

    def __post_init__(self):
        if not 0 <= self.value <= 1:
            raise ValueError("Probability must be between 0 and 1")

    @property
    def percentage(self) -> float:
        return self.value * 100

    def to_odds(self) -> "Odds":
        from .odds import Odds

        if self.value == 0:
            raise ValueError("Cannot convert zero probability to odds")
        return Odds(1 / self.value)


@dataclass(frozen=True)
class ProbabilitySet:
    home: Probability
    draw: Probability
    away: Probability

    def __post_init__(self):
        total = self.home.value + self.draw.value + self.away.value
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Probabilities must sum to 1, got {total}")

    @property
    def normalized(self) -> "ProbabilitySet":
        total = self.home.value + self.draw.value + self.away.value
        return ProbabilitySet(
            home=Probability(self.home.value / total),
            draw=Probability(self.draw.value / total),
            away=Probability(self.away.value / total),
        )
