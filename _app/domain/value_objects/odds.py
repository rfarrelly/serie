import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Odds:
    value: float

    def __post_init__(self):
        if self.value <= 1.0:
            raise ValueError("Odds must be greater than 1.0")

    @property
    def implied_probability(self) -> float:
        return 1 / self.value

    @property
    def decimal(self) -> float:
        return self.value

    def to_american(self) -> int:
        if self.value >= 2.0:
            return int((self.value - 1) * 100)
        else:
            return int(-100 / (self.value - 1))


@dataclass(frozen=True)
class OddsSet:
    home: Odds
    draw: Odds
    away: Odds

    @property
    def overround(self) -> float:
        return sum(
            odds.implied_probability for odds in [self.home, self.draw, self.away]
        )

    @property
    def margin(self) -> float:
        return self.overround - 1.0

    def remove_vig(self) -> "OddsSet":
        """Remove bookmaker margin to get fair odds"""
        # Use iterative method
        c, target_overround, accuracy, current_error = 1, 0, 3, 1000
        max_error = (10 ** (-accuracy)) / 2
        odds_values = [self.home.value, self.draw.value, self.away.value]

        while current_error > max_error:
            f = -1 - target_overround
            for o in odds_values:
                f += (1 / o) ** c

            f_dash = 0
            for o in odds_values:
                f_dash += ((1 / o) ** c) * (-math.log(o))

            h = -f / f_dash
            c = c + h

            t = 0
            for o in odds_values:
                t += (1 / o) ** c
            current_error = abs(t - 1 - target_overround)

        fair_odds = [round(o**c, 3) for o in odds_values]
        return OddsSet(
            home=Odds(fair_odds[0]), draw=Odds(fair_odds[1]), away=Odds(fair_odds[2])
        )
