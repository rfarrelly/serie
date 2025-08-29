# application/use_cases/get_team_matches.py
from typing import List

from domain.entities.match import Match
from domain.repositories import MatchRepository
from shared.types.common_types import LeagueName, Season, TeamName


class GetTeamMatchesUseCase:
    """Use case to retrieve historical matches for a team"""

    def __init__(self, match_repository: MatchRepository):
        self._match_repository = match_repository

    def execute(
        self, team: TeamName, league: LeagueName, season: Season
    ) -> List[Match]:
        """Get all matches for a team in a specific league and season"""
        return self._match_repository.get_matches_by_team(team, league, season)
