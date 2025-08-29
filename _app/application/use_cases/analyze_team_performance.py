# application/use_cases/analyze_team_performance.py
from decimal import Decimal
from statistics import mean
from typing import Any, Dict, List

from domain.entities.match import Match
from domain.repositories import MatchRepository
from shared.types.common_types import BetType, LeagueName, Season, TeamName


class AnalyzeTeamPerformanceUseCase:
    """Use case to analyze team performance metrics"""

    def __init__(self, match_repository: MatchRepository):
        self._match_repository = match_repository

    def execute(
        self, team: TeamName, league: LeagueName, season: Season
    ) -> Dict[str, Any]:
        """Analyze team performance for a season"""
        matches = self._match_repository.get_matches_by_team(team, league, season)
        completed_matches = [m for m in matches if m.result is not None]

        if not completed_matches:
            return {"error": "No completed matches found"}

        # Calculate basic stats
        home_matches = [m for m in completed_matches if m.home_team == team]
        away_matches = [m for m in completed_matches if m.away_team == team]

        total_points = self._calculate_points(completed_matches, team)
        games_played = len(completed_matches)

        # Goals statistics
        goals_for = self._calculate_goals_for(completed_matches, team)
        goals_against = self._calculate_goals_against(completed_matches, team)

        # Form analysis (last 5 matches)
        recent_form = (
            self._analyze_recent_form(completed_matches[-5:], team)
            if len(completed_matches) >= 5
            else []
        )

        # PPI analysis if available
        ppi_analysis = self._analyze_ppi_data(completed_matches)

        return {
            "team": team,
            "league": league,
            "season": season,
            "games_played": games_played,
            "points": total_points,
            "points_per_game": round(total_points / games_played, 2),
            "home_games": len(home_matches),
            "away_games": len(away_matches),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_difference": goals_for - goals_against,
            "recent_form": recent_form,
            "ppi_analysis": ppi_analysis,
        }

    def _calculate_points(self, matches: List[Match], team: TeamName) -> int:
        """Calculate total points earned"""
        points = 0
        for match in matches:
            outcome = match.get_outcome()
            if (match.home_team == team and outcome == BetType.HOME) or (
                match.away_team == team and outcome == BetType.AWAY
            ):
                points += 3
            elif outcome == BetType.DRAW:
                points += 1
        return points

    def _calculate_goals_for(self, matches: List[Match], team: TeamName) -> int:
        """Calculate total goals scored"""
        goals = 0
        for match in matches:
            if match.home_team == team:
                goals += match.result.home_goals
            else:
                goals += match.result.away_goals
        return goals

    def _calculate_goals_against(self, matches: List[Match], team: TeamName) -> int:
        """Calculate total goals conceded"""
        goals = 0
        for match in matches:
            if match.home_team == team:
                goals += match.result.away_goals
            else:
                goals += match.result.home_goals
        return goals

    def _analyze_recent_form(self, matches: List[Match], team: TeamName) -> List[str]:
        """Analyze recent form (W/D/L)"""
        form = []
        for match in matches:
            outcome = match.get_outcome()
            if (match.home_team == team and outcome == BetType.HOME) or (
                match.away_team == team and outcome == BetType.AWAY
            ):
                form.append("W")
            elif outcome == BetType.DRAW:
                form.append("D")
            else:
                form.append("L")
        return form

    def _analyze_ppi_data(self, matches: List[Match]) -> Dict[str, Any]:
        """Analyze PPI data if available"""
        ppi_matches = [m for m in matches if m.ppi_data is not None]

        if not ppi_matches:
            return {"available": False}

        home_ppis = [float(m.ppi_data.home_ppi) for m in ppi_matches if m.home_team]
        away_ppis = [float(m.ppi_data.away_ppi) for m in ppi_matches if m.away_team]
        ppi_diffs = [float(m.ppi_data.ppi_difference) for m in ppi_matches]

        return {
            "available": True,
            "matches_with_ppi": len(ppi_matches),
            "average_ppi_difference": round(mean(ppi_diffs), 3) if ppi_diffs else 0,
            "max_ppi_difference": round(max(ppi_diffs), 3) if ppi_diffs else 0,
            "min_ppi_difference": round(min(ppi_diffs), 3) if ppi_diffs else 0,
        }
