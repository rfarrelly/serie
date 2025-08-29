from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from domain.entities.match import Match
from shared.types.common_types import LeagueName, Season, TeamName


class MatchRepository(ABC):
    """Domain interface - doesn't know about CSV/pandas"""

    @abstractmethod
    def get_matches_by_team(
        self, team: TeamName, league: LeagueName, season: Season
    ) -> List[Match]:
        pass

    @abstractmethod
    def get_matches_by_league_and_season(
        self, league: LeagueName, season: Season
    ) -> List[Match]:
        pass
