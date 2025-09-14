from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from .entities import Match, Team


class MatchRepository(ABC):
    @abstractmethod
    def get_by_date_range(self, start: datetime, end: datetime) -> List[Match]:
        pass

    @abstractmethod
    def get_by_league(self, league: str) -> List[Match]:
        pass

    @abstractmethod
    def get_historical_matches(self, league: str, seasons: List[str]) -> List[Match]:
        pass

    @abstractmethod
    def save_matches(self, matches: List[Match]) -> None:
        pass


class TeamRepository(ABC):
    @abstractmethod
    def get_by_league(self, league: str) -> List[Team]:
        pass

    @abstractmethod
    def get_team_mapping(self) -> dict:
        pass
