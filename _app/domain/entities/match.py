from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..value_objects.odds import Odds


@dataclass
class Match:
    home_team: str
    away_team: str
    date: datetime
    league: str
    season: str
    week: Optional[int] = None
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    odds: Optional[Odds] = None

    @property
    def is_played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def result(self) -> Optional["MatchResult"]:
        if not self.is_played:
            return None
        if self.home_goals > self.away_goals:
            return MatchResult.HOME_WIN
        elif self.home_goals < self.away_goals:
            return MatchResult.AWAY_WIN
        return MatchResult.DRAW


from enum import Enum


class MatchResult(Enum):
    HOME_WIN = 0
    DRAW = 1
    AWAY_WIN = 2
