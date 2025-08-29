# domain/services/ppi_calculator.py
"""
Point Performance Index (PPI) calculation service.

Extracted from stats.py - preserves your mathematical expertise
while providing clean, testable domain service.
"""

from dataclasses import dataclass
from decimal import Decimal
from statistics import mean
from typing import Dict, List, Tuple

from domain.entities.match import Match
from shared.types.common_types import BetType, TeamName


@dataclass(frozen=True)
class PPICalculationResult:
    """Result of PPI calculation for a team"""

    team: TeamName
    points_per_game: Decimal
    opponent_average_ppg: Decimal
    ppi_value: Decimal
    matches_analyzed: int
    home_matches: int
    away_matches: int


class PPICalculator:
    """
    Domain service for Point Performance Index calculations.

    PPI = Team's PPG × Average PPG of opponents faced

    This measures team performance adjusted for opponent strength.
    """

    def calculate_team_ppi(
        self,
        target_team: TeamName,
        matches: List[Match],
        include_match_details: bool = False,
    ) -> PPICalculationResult:
        """
        Calculate PPI for a specific team across a set of matches.

        Args:
            target_team: Team to analyze
            matches: Historical matches to analyze
            include_match_details: Whether to include detailed match breakdown

        Returns:
            PPICalculationResult with calculated metrics
        """
        # Filter matches involving the target team
        team_matches = [
            m
            for m in matches
            if (m.home_team == target_team or m.away_team == target_team)
            and m.result is not None
        ]

        if not team_matches:
            return PPICalculationResult(
                team=target_team,
                points_per_game=Decimal("0"),
                opponent_average_ppg=Decimal("0"),
                ppi_value=Decimal("0"),
                matches_analyzed=0,
                home_matches=0,
                away_matches=0,
            )

        # Calculate team's points per game
        team_ppg = self._calculate_team_points_per_game(target_team, team_matches)

        # Calculate opponent strength (average PPG of opponents faced)
        opponent_avg_ppg = self._calculate_opponent_average_ppg(
            target_team, team_matches, matches
        )

        # Calculate PPI: Team PPG × Opponent Average PPG
        ppi_value = team_ppg * opponent_avg_ppg

        # Count match types
        home_matches = len([m for m in team_matches if m.home_team == target_team])
        away_matches = len(team_matches) - home_matches

        return PPICalculationResult(
            team=target_team,
            points_per_game=team_ppg,
            opponent_average_ppg=opponent_avg_ppg,
            ppi_value=ppi_value,
            matches_analyzed=len(team_matches),
            home_matches=home_matches,
            away_matches=away_matches,
        )

    def calculate_ppi_differential(
        self, home_team: TeamName, away_team: TeamName, matches: List[Match]
    ) -> Tuple[PPICalculationResult, PPICalculationResult, Decimal]:
        """
        Calculate PPI for both teams and return the differential.

        Returns:
            (home_ppi_result, away_ppi_result, ppi_differential)
        """
        home_ppi = self.calculate_team_ppi(home_team, matches)
        away_ppi = self.calculate_team_ppi(away_team, matches)

        differential = abs(home_ppi.ppi_value - away_ppi.ppi_value)

        return home_ppi, away_ppi, differential

    def _calculate_team_points_per_game(
        self, team: TeamName, team_matches: List[Match]
    ) -> Decimal:
        """Calculate points per game for a team"""
        total_points = 0

        for match in team_matches:
            outcome = match.get_outcome()

            # Award points based on outcome and home/away status
            if team == match.home_team:
                if outcome == BetType.HOME:
                    total_points += 3  # Win at home
                elif outcome == BetType.DRAW:
                    total_points += 1  # Draw at home
                # Loss = 0 points
            else:  # team == match.away_team
                if outcome == BetType.AWAY:
                    total_points += 3  # Win away
                elif outcome == BetType.DRAW:
                    total_points += 1  # Draw away
                # Loss = 0 points

        games_played = len(team_matches)
        ppg = Decimal(str(total_points)) / Decimal(str(games_played))

        return ppg.quantize(Decimal("0.001"))  # Round to 3 decimal places

    def _calculate_opponent_average_ppg(
        self, target_team: TeamName, team_matches: List[Match], all_matches: List[Match]
    ) -> Decimal:
        """
        Calculate average PPG of all opponents faced by the target team.

        This is the strength of schedule component of PPI.
        """
        opponents_faced = set()

        # Collect all opponents
        for match in team_matches:
            if match.home_team == target_team:
                opponents_faced.add(match.away_team)
            else:
                opponents_faced.add(match.home_team)

        if not opponents_faced:
            return Decimal("0")

        # Calculate PPG for each opponent
        opponent_ppgs = []

        for opponent in opponents_faced:
            opponent_ppg = self._calculate_team_points_per_game(
                opponent,
                [
                    m
                    for m in all_matches
                    if (m.home_team == opponent or m.away_team == opponent)
                    and m.result is not None
                ],
            )
            opponent_ppgs.append(float(opponent_ppg))

        # Return average PPG of all opponents
        avg_opponent_ppg = Decimal(str(mean(opponent_ppgs)))
        return avg_opponent_ppg.quantize(Decimal("0.001"))

    def calculate_league_ppi_rankings(
        self, matches: List[Match], league_filter: str = None
    ) -> List[PPICalculationResult]:
        """
        Calculate PPI for all teams in a league and return ranked list.
        """
        # Get all teams from the matches
        all_teams = set()
        filtered_matches = matches

        if league_filter:
            filtered_matches = [m for m in matches if m.league == league_filter]

        for match in filtered_matches:
            all_teams.add(match.home_team)
            all_teams.add(match.away_team)

        # Calculate PPI for each team
        team_ppis = []
        for team in all_teams:
            ppi_result = self.calculate_team_ppi(team, filtered_matches)
            team_ppis.append(ppi_result)

        # Sort by PPI value (descending)
        return sorted(team_ppis, key=lambda x: x.ppi_value, reverse=True)

    def analyze_ppi_trends(
        self, team: TeamName, matches: List[Match], window_size: int = 5
    ) -> List[Tuple[int, Decimal]]:
        """
        Analyze PPI trends over time using a rolling window.

        Returns:
            List of (match_number, ppi_value) tuples showing trend
        """
        team_matches = [
            m
            for m in sorted(matches, key=lambda x: x.date)
            if (m.home_team == team or m.away_team == team) and m.result is not None
        ]

        if len(team_matches) < window_size:
            return []

        ppi_trends = []

        for i in range(window_size, len(team_matches) + 1):
            window_matches = team_matches[:i]  # All matches up to this point
            recent_matches = team_matches[max(0, i - window_size) : i]  # Recent window

            # Calculate PPI using all historical data for opponent strength
            # but only recent matches for team performance
            team_ppg = self._calculate_team_points_per_game(team, recent_matches)
            opponent_avg_ppg = self._calculate_opponent_average_ppg(
                team, recent_matches, window_matches
            )

            ppi_value = team_ppg * opponent_avg_ppg
            ppi_trends.append((i, ppi_value))

        return ppi_trends
