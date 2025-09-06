from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..entities.team import Team, TeamRatings


class TeamRepository(ABC):
    @abstractmethod
    def find_by_name(self, name: str) -> Optional[Team]:
        pass

    @abstractmethod
    def find_by_league(self, league: str) -> List[Team]:
        pass

    @abstractmethod
    def find_all(self) -> List[Team]:
        pass

    @abstractmethod
    def save(self, team: Team) -> None:
        pass

    @abstractmethod
    def get_name_mappings(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def save_name_mappings(self, mappings: Dict[str, str]) -> None:
        pass


class TeamRatingsRepository(ABC):
    @abstractmethod
    def find_by_team(self, team_name: str) -> Optional[TeamRatings]:
        pass

    @abstractmethod
    def find_all(self) -> List[TeamRatings]:
        pass

    @abstractmethod
    def save(self, ratings: TeamRatings) -> None:
        pass

    @abstractmethod
    def save_all(self, ratings: List[TeamRatings]) -> None:
        pass
