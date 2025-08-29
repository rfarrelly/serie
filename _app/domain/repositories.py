# domain/repositories.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from domain.entities.match import Fixture, Match
from shared.types.common_types import LeagueName, Season, TeamName


class MatchRepository(ABC):
    """Domain interface for accessing historical match data"""

    @abstractmethod
    def get_matches_by_team(
        self, team: TeamName, league: LeagueName, season: Season
    ) -> List[Match]:
        """Get all matches for a specific team in a league and season"""
        pass

    @abstractmethod
    def get_matches_by_league_and_season(
        self, league: LeagueName, season: Season
    ) -> List[Match]:
        """Get all matches for a league and season"""
        pass

    @abstractmethod
    def get_recent_matches(self, team: TeamName, num_matches: int = 10) -> List[Match]:
        """Get the most recent matches for a team"""
        pass


class FixtureRepository(ABC):
    """Domain interface for accessing upcoming fixture data"""

    @abstractmethod
    def get_upcoming_fixtures(
        self,
        league: Optional[LeagueName] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> List[Fixture]:
        """Get upcoming fixtures with optional filters"""
        pass

    @abstractmethod
    def get_fixtures_with_predictions(self) -> List[Fixture]:
        """Get fixtures that have model predictions"""
        pass

    @abstractmethod
    def get_betting_candidates(self, min_edge: float = 0.02) -> List[Fixture]:
        """Get fixtures identified as betting candidates"""
        pass
