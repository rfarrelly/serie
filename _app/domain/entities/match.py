from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from shared.types.common_types import BetType, LeagueName, TeamName


@dataclass
class MatchResult:
    """Value object for match result"""

    home_goals: int
    away_goals: int

    def __post_init__(self):
        if self.home_goals < 0 or self.away_goals < 0:
            raise ValueError("Goals cannot be negative")


class Match:
    """Match entity - represents a completed football match"""

    def __init__(
        self,
        home_team: TeamName,
        away_team: TeamName,
        league: LeagueName,
        date: datetime,
        week: int,
    ):
        self._home_team = home_team
        self._away_team = away_team
        self._league = league
        self._date = date
        self._week = week
        self._result: Optional[MatchResult] = None

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
    def result(self) -> Optional[MatchResult]:
        return self._result

    def record_result(self, home_goals: int, away_goals: int) -> None:
        """Business rule: Can only record result once"""
        if self._result is not None:
            raise ValueError("Match result already recorded")

        self._result = MatchResult(home_goals, away_goals)

    def get_outcome(self) -> BetType:
        """Domain logic: Determine match outcome"""
        if not self._result:
            raise ValueError("Match not completed")

        if self._result.home_goals > self._result.away_goals:
            return BetType.HOME
        elif self._result.home_goals < self._result.away_goals:
            return BetType.AWAY
        else:
            return BetType.DRAW


@dataclass
class Fixture:
    """Represents an upcoming match (not yet played)"""

    home_team: TeamName
    away_team: TeamName
    league: LeagueName
    date: datetime
    week: Optional[int] = None
