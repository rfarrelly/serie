# domain/entities/betting.py
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from shared.exceptions import InvalidOddsException, InvalidProbabilityException
from shared.types.common_types import BetType, BookmakerType


@dataclass(frozen=True)
class Odds:
    """Immutable odds representation with validation"""

    decimal: Decimal

    def __post_init__(self):
        if self.decimal <= 1.0:
            raise InvalidOddsException(f"Odds must be > 1.0, got {self.decimal}")

    @property
    def implied_probability(self) -> Decimal:
        """Convert odds to implied probability"""
        return Decimal("1.0") / self.decimal

    @classmethod
    def from_float(cls, value: Optional[float]) -> Optional["Odds"]:
        """Safe constructor from float (handles NaN/None from CSV)"""
        if value is None or value != value:  # NaN check
            return None
        return cls(Decimal(str(value)))


@dataclass(frozen=True)
class MarketOdds:
    """Complete odds set for a match from a specific bookmaker"""

    home: Odds
    draw: Odds
    away: Odds
    bookmaker: BookmakerType

    @property
    def overround(self) -> Decimal:
        """Calculate bookmaker's overround (profit margin)"""
        return (
            self.home.implied_probability
            + self.draw.implied_probability
            + self.away.implied_probability
        ) - Decimal("1.0")

    @classmethod
    def from_csv_row(cls, row, bookmaker: BookmakerType) -> Optional["MarketOdds"]:
        """Create MarketOdds from your CSV row data"""
        suffix = bookmaker.value

        home_odds = Odds.from_float(row.get(f"{suffix}H"))
        draw_odds = Odds.from_float(row.get(f"{suffix}D"))
        away_odds = Odds.from_float(row.get(f"{suffix}A"))

        if not all([home_odds, draw_odds, away_odds]):
            return None

        return cls(home=home_odds, draw=draw_odds, away=away_odds, bookmaker=bookmaker)


@dataclass(frozen=True)
class Probability:
    """Probability value object with validation"""

    value: Decimal

    def __post_init__(self):
        if not (0 <= self.value <= 1):
            raise InvalidProbabilityException(
                f"Probability must be 0-1, got {self.value}"
            )

    @classmethod
    def from_float(cls, value: Optional[float]) -> Optional["Probability"]:
        """Safe constructor from float"""
        if value is None or value != value:
            return None
        return cls(Decimal(str(value)))


@dataclass(frozen=True)
class ModelProbabilities:
    """Complete probability set for match outcomes from a specific model"""

    home_win: Probability
    draw: Probability
    away_win: Probability
    model_type: str

    def __post_init__(self):
        total = self.home_win.value + self.draw.value + self.away_win.value
        if abs(total - Decimal("1.0")) > Decimal("0.001"):
            raise InvalidProbabilityException(
                f"Probabilities must sum to 1.0, got {total}"
            )

    @classmethod
    def from_csv_row(cls, row, model_prefix: str) -> Optional["ModelProbabilities"]:
        """Create ModelProbabilities from your CSV data"""
        home_prob = Probability.from_float(row.get(f"{model_prefix}_Prob_H"))
        draw_prob = Probability.from_float(row.get(f"{model_prefix}_Prob_D"))
        away_prob = Probability.from_float(row.get(f"{model_prefix}_Prob_A"))

        if not all([home_prob, draw_prob, away_prob]):
            return None

        return cls(
            home_win=home_prob,
            draw=draw_prob,
            away_win=away_prob,
            model_type=model_prefix,
        )


@dataclass(frozen=True)
class PPIData:
    """Point Performance Index data"""

    home_ppi: Decimal
    away_ppi: Decimal
    ppi_difference: Decimal
    home_ppg: Optional[Decimal] = None
    away_ppg: Optional[Decimal] = None
    home_opp_ppg: Optional[Decimal] = None
    away_opp_ppg: Optional[Decimal] = None

    @classmethod
    def from_csv_row(cls, row) -> Optional["PPIData"]:
        """Create PPIData from your CSV row"""
        try:
            return cls(
                home_ppi=Decimal(str(row["hPPI"])),
                away_ppi=Decimal(str(row["aPPI"])),
                ppi_difference=Decimal(str(row["PPI_Diff"])),
                home_ppg=(
                    Decimal(str(row["hPPG"]))
                    if "hPPG" in row and row["hPPG"] is not None
                    else None
                ),
                away_ppg=(
                    Decimal(str(row["aPPG"]))
                    if "aPPG" in row and row["aPPG"] is not None
                    else None
                ),
                home_opp_ppg=(
                    Decimal(str(row["hOppPPG"]))
                    if "hOppPPG" in row and row["hOppPPG"] is not None
                    else None
                ),
                away_opp_ppg=(
                    Decimal(str(row["aOppPPG"]))
                    if "aOppPPG" in row and row["aOppPPG"] is not None
                    else None
                ),
            )
        except (ValueError, KeyError):
            return None


@dataclass(frozen=True)
class BettingOpportunity:
    """Represents a calculated betting opportunity"""

    bet_type: BetType
    edge: Decimal
    model_probability: Probability
    market_probability: Probability
    recommended_odds: Odds
    fair_odds: Odds
    expected_value: Decimal
    kelly_fraction: Optional[Decimal] = None

    def is_significant(self, threshold: Decimal = Decimal("0.02")) -> bool:
        """Domain rule: What constitutes a significant betting edge"""
        return self.edge >= threshold

    @classmethod
    def from_csv_row(cls, row) -> Optional["BettingOpportunity"]:
        """Create BettingOpportunity from your enhanced CSV data"""
        try:
            bet_type_str = row.get("Bet_Type")
            if not bet_type_str:
                return None

            bet_type = BetType(bet_type_str)

            return cls(
                bet_type=bet_type,
                edge=Decimal(str(row["Edge"])),
                model_probability=Probability(Decimal(str(row["Model_Prob"]))),
                market_probability=Probability(Decimal(str(row["Market_Prob"]))),
                recommended_odds=Odds(Decimal(str(row["Soft_Odds"]))),
                fair_odds=Odds(Decimal(str(row["Fair_Odds_Selected"]))),
                expected_value=Decimal(
                    str(row[f"EV_{bet_type.value[0]}"])
                ),  # EV_H, EV_D, EV_A
            )
        except (
            ValueError,
            KeyError,
            InvalidOddsException,
            InvalidProbabilityException,
        ):
            return None
