from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Team:
    name: str
    league: str

    def __hash__(self):
        return hash((self.name, self.league))


@dataclass
class Match:
    id: str
    date: datetime
    home_team: Team
    away_team: Team
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    week: Optional[int] = None

    @property
    def is_played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def result(self) -> Optional[str]:
        if not self.is_played:
            return None
        if self.home_goals > self.away_goals:
            return "H"
        elif self.home_goals < self.away_goals:
            return "A"
        else:
            return "D"
