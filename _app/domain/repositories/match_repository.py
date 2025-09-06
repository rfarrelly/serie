from abc import ABC, abstractmethod
from typing import List

from ..entities.match import Match
from ..value_objects.timeframe import Season, Timeframe


class MatchRepository(ABC):
    @abstractmethod
    def find_by_teams(self, home_team: str, away_team: str) -> List[Match]:
        pass

    @abstractmethod
    def find_by_league_and_season(self, league: str, season: Season) -> List[Match]:
        pass

    @abstractmethod
    def find_by_timeframe(self, timeframe: Timeframe) -> List[Match]:
        pass

    @abstractmethod
    def find_played_matches(self) -> List[Match]:
        pass

    @abstractmethod
    def find_upcoming_matches(self) -> List[Match]:
        pass

    @abstractmethod
    def save(self, match: Match) -> None:
        pass

    @abstractmethod
    def save_all(self, matches: List[Match]) -> None:
        pass
