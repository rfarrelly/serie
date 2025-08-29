# interfaces/cli/commands/data_commands.py
"""
Data management commands
"""

from pathlib import Path
from typing import List

from domain.services.ppi_calculator import PPICalculator
from infrastructure.data.repositories.csv_match_repository import CSVMatchRepository
from shared.types.common_types import LeagueName, Season


class AnalyzeCommand:
    """
    Command for analyzing team/league performance using domain services
    """

    def __init__(self):
        self.name = "analyze"
        self.description = (
            "Analyze team or league performance using PPI and domain services"
        )
        self.help_text = """
Analyze football performance using domain services:

Usage:
  analyze team TEAM --league LEAGUE --season SEASON
  analyze league LEAGUE --season SEASON  
  analyze rankings LEAGUE --season SEASON

Examples:
  analyze team Arsenal --league Premier-League --season 2023-2024
  analyze league Premier-League --season 2023-2024
  analyze rankings Premier-League --season 2023-2024
        """

    def handle(self, args: List[str]) -> int:
        """Handle analyze command"""

        try:
            if not args:
                print(self.help_text)
                return 0

            subcommand = args[0]

            if subcommand == "team":
                return self._analyze_team(args[1:])
            elif subcommand == "league":
                return self._analyze_league(args[1:])
            elif subcommand == "rankings":
                return self._analyze_rankings(args[1:])
            else:
                print(f"Unknown analyze subcommand: {subcommand}")
                print(self.help_text)
                return 1

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return 1

    def _analyze_team(self, args: List[str]) -> int:
        """Analyze specific team performance"""

        # Parse arguments
        if len(args) < 5 or "--league" not in args or "--season" not in args:
            print("Usage: analyze team TEAM --league LEAGUE --season SEASON")
            return 1

        team_name = args[0]
        league_idx = args.index("--league")
        season_idx = args.index("--season")

        league = LeagueName(args[league_idx + 1])
        season = Season(args[season_idx + 1])

        print(f"📊 Analyzing {team_name} in {league} ({season})")
        print("=" * 50)

        # Use domain services
        match_repo = CSVMatchRepository()
        ppi_calculator = PPICalculator()

        # Get matches and calculate PPI
        matches = match_repo.get_matches_by_league_and_season(league, season)
        team_ppi = ppi_calculator.calculate_team_ppi(team_name, matches)

        # Display results
        print(f"Team: {team_ppi.team}")
        print(f"Matches Analyzed: {team_ppi.matches_analyzed}")
        print(f"Home/Away Split: {team_ppi.home_matches}/{team_ppi.away_matches}")
        print(f"Points Per Game: {team_ppi.points_per_game:.3f}")
        print(f"Opponent Strength: {team_ppi.opponent_average_ppg:.3f}")
        print(f"PPI Value: {team_ppi.ppi_value:.3f}")

        return 0

    def _analyze_league(self, args: List[str]) -> int:
        """Analyze league overview"""

        if len(args) < 3 or "--season" not in args:
            print("Usage: analyze league LEAGUE --season SEASON")
            return 1

        league = LeagueName(args[0])
        season_idx = args.index("--season")
        season = Season(args[season_idx + 1])

        print(f"📊 League Analysis: {league} ({season})")
        print("=" * 50)

        # Use domain services
        match_repo = CSVMatchRepository()
        matches = match_repo.get_matches_by_league_and_season(league, season)

        print(f"Total Matches: {len(matches)}")

        teams = set()
        for match in matches:
            teams.add(match.home_team)
            teams.add(match.away_team)

        print(f"Teams: {len(teams)}")

        if matches:
            completed = [m for m in matches if m.result is not None]
            print(f"Completed Matches: {len(completed)}")

            total_goals = sum(
                m.result.home_goals + m.result.away_goals for m in completed
            )
            avg_goals = total_goals / len(completed) if completed else 0
            print(f"Average Goals per Match: {avg_goals:.2f}")

        return 0

    def _analyze_rankings(self, args: List[str]) -> int:
        """Show PPI rankings for league"""

        if len(args) < 3 or "--season" not in args:
            print("Usage: analyze rankings LEAGUE --season SEASON")
            return 1

        league = LeagueName(args[0])
        season_idx = args.index("--season")
        season = Season(args[season_idx + 1])

        print(f"📊 PPI Rankings: {league} ({season})")
        print("=" * 50)

        # Use domain services
        match_repo = CSVMatchRepository()
        ppi_calculator = PPICalculator()

        matches = match_repo.get_matches_by_league_and_season(league, season)
        rankings = ppi_calculator.calculate_league_ppi_rankings(matches, league)

        print(
            f"{'Rank':<4} {'Team':<20} {'PPI':<8} {'PPG':<6} {'Opp Str':<8} {'Matches':<8}"
        )
        print("-" * 60)

        for i, team_ppi in enumerate(rankings[:20], 1):
            print(
                f"{i:<4} {team_ppi.team:<20} {team_ppi.ppi_value:<8.3f} "
                f"{team_ppi.points_per_game:<6.3f} {team_ppi.opponent_average_ppg:<8.3f} "
                f"{team_ppi.matches_analyzed:<8}"
            )

        return 0
