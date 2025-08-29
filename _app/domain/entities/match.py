# domain/entities/match.py
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
from domain.entities.betting import (
    BettingOpportunity,
    MarketOdds,
    ModelProbabilities,
    PPIData,
)
from shared.exceptions import MatchAlreadyCompletedException, MatchNotCompletedException
from shared.types.common_types import (
    BetType,
    BookmakerType,
    LeagueName,
    Season,
    TeamName,
)


@dataclass(frozen=True)
class MatchResult:
    """Value object for match result"""

    home_goals: int
    away_goals: int

    def __post_init__(self):
        if self.home_goals < 0 or self.away_goals < 0:
            raise ValueError("Goals cannot be negative")

    @property
    def goal_difference(self) -> int:
        """Home team perspective goal difference"""
        return self.home_goals - self.away_goals


class Match:
    """Match entity - represents a completed football match"""

    def __init__(
        self,
        home_team: TeamName,
        away_team: TeamName,
        league: LeagueName,
        date: datetime,
        week: int,
        season: Optional[Season] = None,
        day_of_week: Optional[str] = None,
    ):
        self._home_team = home_team
        self._away_team = away_team
        self._league = league
        self._date = date
        self._week = week
        self._season = season
        self._day_of_week = day_of_week
        self._result: Optional[MatchResult] = None
        self._market_odds: Dict[BookmakerType, MarketOdds] = {}
        self._ppi_data: Optional[PPIData] = None

    # Properties
    @property
    def home_team(self) -> TeamName:
        return self._home_team

    @property
    def away_team(self) -> TeamName:
        return self._away_team

    @property
    def league(self) -> LeagueName:
        return self._league

    @property
    def date(self) -> datetime:
        return self._date

    @property
    def week(self) -> int:
        return self._week

    @property
    def season(self) -> Optional[Season]:
        return self._season

    @property
    def result(self) -> Optional[MatchResult]:
        return self._result

    @property
    def market_odds(self) -> Dict[BookmakerType, MarketOdds]:
        return self._market_odds.copy()

    @property
    def ppi_data(self) -> Optional[PPIData]:
        return self._ppi_data

    # Business logic methods
    def record_result(self, home_goals: int, away_goals: int) -> None:
        """Business rule: Can only record result once"""
        if self._result is not None:
            raise MatchAlreadyCompletedException("Match result already recorded")

        self._result = MatchResult(home_goals, away_goals)

    def get_outcome(self) -> BetType:
        """Domain logic: Determine match outcome"""
        if not self._result:
            raise MatchNotCompletedException("Match not completed")

        if self._result.home_goals > self._result.away_goals:
            return BetType.HOME
        elif self._result.home_goals < self._result.away_goals:
            return BetType.AWAY
        else:
            return BetType.DRAW

    def add_market_odds(self, odds: MarketOdds) -> None:
        """Add odds from a specific bookmaker"""
        self._market_odds[odds.bookmaker] = odds

    def set_ppi_data(self, ppi_data: PPIData) -> None:
        """Set PPI analysis data"""
        self._ppi_data = ppi_data

    def get_sharp_odds(self) -> Optional[MarketOdds]:
        """Get sharp (Pinnacle) odds if available"""
        return self._market_odds.get(BookmakerType.PINNACLE)

    def get_soft_odds(self) -> Optional[MarketOdds]:
        """Get soft (Bet365) odds if available"""
        return self._market_odds.get(BookmakerType.BET365)

    @classmethod
    def from_historical_csv_row(cls, row) -> "Match":
        """Create Match from your historical_ppi_and_odds.csv data"""
        match = cls(
            home_team=TeamName(row["Home"]),
            away_team=TeamName(row["Away"]),
            league=LeagueName(row["League"]),
            date=pd.to_datetime(row["Date"]),
            week=int(row["Wk"]),
            season=Season(row["Season"]) if "Season" in row else None,
            day_of_week=row.get("Day"),
        )

        # Add result if available
        if (
            "FTHG" in row
            and "FTAG" in row
            and pd.notna(row["FTHG"])
            and pd.notna(row["FTAG"])
        ):
            match.record_result(int(row["FTHG"]), int(row["FTAG"]))

        # Add market odds
        for bookmaker in [BookmakerType.PINNACLE, BookmakerType.BET365]:
            odds = MarketOdds.from_csv_row(row, bookmaker)
            if odds:
                match.add_market_odds(odds)

        # Add PPI data
        ppi_data = PPIData.from_csv_row(row)
        if ppi_data:
            match.set_ppi_data(ppi_data)

        return match


class Fixture:
    """Represents an upcoming match (not yet played) with predictions"""

    def __init__(
        self,
        home_team: TeamName,
        away_team: TeamName,
        league: LeagueName,
        date: datetime,
        week: Optional[int] = None,
    ):
        self._home_team = home_team
        self._away_team = away_team
        self._league = league
        self._date = date
        self._week = week
        self._market_odds: Dict[BookmakerType, MarketOdds] = {}
        self._model_predictions: Dict[str, ModelProbabilities] = {}
        self._ppi_data: Optional[PPIData] = None
        self._betting_opportunities: List[BettingOpportunity] = []

    # Properties
    @property
    def home_team(self) -> TeamName:
        return self._home_team

    @property
    def away_team(self) -> TeamName:
        return self._away_team

    @property
    def league(self) -> LeagueName:
        return self._league

    @property
    def date(self) -> datetime:
        return self._date

    @property
    def week(self) -> Optional[int]:
        return self._week

    @property
    def market_odds(self) -> Dict[BookmakerType, MarketOdds]:
        return self._market_odds.copy()

    @property
    def model_predictions(self) -> Dict[str, ModelProbabilities]:
        return self._model_predictions.copy()

    @property
    def ppi_data(self) -> Optional[PPIData]:
        return self._ppi_data

    @property
    def betting_opportunities(self) -> List[BettingOpportunity]:
        return self._betting_opportunities.copy()

    # Business logic methods
    def add_market_odds(self, odds: MarketOdds) -> None:
        """Add odds from a specific bookmaker"""
        self._market_odds[odds.bookmaker] = odds

    def add_model_prediction(self, prediction: ModelProbabilities) -> None:
        """Add prediction from a specific model"""
        self._model_predictions[prediction.model_type] = prediction

    def set_ppi_data(self, ppi_data: PPIData) -> None:
        """Set PPI analysis data"""
        self._ppi_data = ppi_data

    def add_betting_opportunity(self, opportunity: BettingOpportunity) -> None:
        """Add a calculated betting opportunity"""
        self._betting_opportunities.append(opportunity)

    def get_significant_opportunities(
        self, min_edge: Decimal = Decimal("0.02")
    ) -> List[BettingOpportunity]:
        """Get betting opportunities above minimum edge threshold"""
        return [
            opp for opp in self._betting_opportunities if opp.is_significant(min_edge)
        ]

    def has_model_prediction(self, model_type: str) -> bool:
        """Check if fixture has prediction from specific model"""
        return model_type in self._model_predictions

    @classmethod
    def from_enhanced_csv_row(cls, row) -> "Fixture":
        """Create Fixture from your latest_zsd_enhanced.csv data"""
        fixture = cls(
            home_team=TeamName(row["Home"]),
            away_team=TeamName(row["Away"]),
            league=LeagueName(row["League"]),
            date=pd.to_datetime(row["Date"]),
            week=int(row["Wk"]) if pd.notna(row.get("Wk")) else None,
        )

        # Add market odds
        for bookmaker in [BookmakerType.PINNACLE, BookmakerType.BET365]:
            odds = MarketOdds.from_csv_row(row, bookmaker)
            if odds:
                fixture.add_market_odds(odds)

        # Add model predictions
        for model_prefix in ["Poisson", "ZIP", "MOV", "ZSD"]:
            prediction = ModelProbabilities.from_csv_row(row, model_prefix)
            if prediction:
                fixture.add_model_prediction(prediction)

        # Add PPI data
        ppi_data = PPIData.from_csv_row(row)
        if ppi_data:
            fixture.set_ppi_data(ppi_data)

        # Add betting opportunity if it exists
        opportunity = BettingOpportunity.from_csv_row(row)
        if opportunity:
            fixture.add_betting_opportunity(opportunity)

        return fixture

    @classmethod
    def from_ppi_csv_row(cls, row) -> "Fixture":
        """Create Fixture from your fixtures_ppi_and_odds.csv data"""
        fixture = cls(
            home_team=TeamName(row["Home"]),
            away_team=TeamName(row["Away"]),
            league=LeagueName(row["League"]),
            date=pd.to_datetime(row["Date"]),
            week=int(row["Wk"]) if pd.notna(row.get("Wk")) else None,
        )

        # Add market odds
        for bookmaker in [BookmakerType.PINNACLE, BookmakerType.BET365]:
            odds = MarketOdds.from_csv_row(row, bookmaker)
            if odds:
                fixture.add_market_odds(odds)

        # Add PPI data
        ppi_data = PPIData.from_csv_row(row)
        if ppi_data:
            fixture.set_ppi_data(ppi_data)

        return fixture
