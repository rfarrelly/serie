from typing import List

import numpy as np

from ..entities.match import Match
from ..entities.ppi import PPIData


class PPIService:
    """Service for calculating Points Performance Index"""

    def __init__(self):
        self._team_ppg_cache = {}

    def calculate_team_ppi(self, team_name: str, matches: List[Match]) -> PPIData:
        """Calculate PPI for a specific team"""
        team_matches = [
            m
            for m in matches
            if m.is_played and (m.home_team == team_name or m.away_team == team_name)
        ]

        if not team_matches:
            return PPIData(team_name, "Unknown", 0.0, 0.0, 0.0)

        # Calculate team's points
        team_points = []
        opponent_teams = []

        for match in team_matches:
            if match.home_team == team_name:
                # Team playing at home
                if match.home_goals > match.away_goals:
                    team_points.append(3)
                elif match.home_goals == match.away_goals:
                    team_points.append(1)
                else:
                    team_points.append(0)
                opponent_teams.append(match.away_team)
            else:
                # Team playing away
                if match.away_goals > match.home_goals:
                    team_points.append(3)
                elif match.away_goals == match.home_goals:
                    team_points.append(1)
                else:
                    team_points.append(0)
                opponent_teams.append(match.home_team)

        # Calculate PPG
        ppg = np.mean(team_points) if team_points else 0.0

        # Calculate opponent PPG using cached values to avoid recursion
        opp_ppg = self._calculate_opponent_ppg(
            opponent_teams, matches, exclude_team=team_name
        )

        # Calculate PPI
        ppi = ppg * opp_ppg

        league = team_matches[0].league if team_matches else "Unknown"

        return PPIData(team_name, league, ppg, opp_ppg, ppi)

    def _calculate_opponent_ppg(
        self, opponent_teams: List[str], all_matches: List[Match], exclude_team: str
    ) -> float:
        """Calculate average PPG of opponent teams, excluding the target team"""
        if not opponent_teams:
            return 0.0

        ppg_values = []
        for opp_team in set(opponent_teams):
            if opp_team == exclude_team:
                continue

            # Calculate PPG for this opponent (excluding matches against the target team)
            opp_matches = [
                m
                for m in all_matches
                if m.is_played
                and (m.home_team == opp_team or m.away_team == opp_team)
                and not (m.home_team == exclude_team or m.away_team == exclude_team)
            ]

            if not opp_matches:
                continue

            opp_points = []
            for match in opp_matches:
                if match.home_team == opp_team:
                    if match.home_goals > match.away_goals:
                        opp_points.append(3)
                    elif match.home_goals == match.away_goals:
                        opp_points.append(1)
                    else:
                        opp_points.append(0)
                else:
                    if match.away_goals > match.home_goals:
                        opp_points.append(3)
                    elif match.away_goals == match.home_goals:
                        opp_points.append(1)
                    else:
                        opp_points.append(0)

            if opp_points:
                ppg_values.append(np.mean(opp_points))

        return np.mean(ppg_values) if ppg_values else 1.5  # Default PPG

    def calculate_ppi_difference(
        self, home_team: str, away_team: str, matches: List[Match]
    ) -> float:
        """Calculate PPI difference between home and away teams"""
        home_ppi = self.calculate_team_ppi(home_team, matches)
        away_ppi = self.calculate_team_ppi(away_team, matches)
        return abs(home_ppi.ppi - away_ppi.ppi)
