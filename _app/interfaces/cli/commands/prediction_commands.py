# interfaces/cli/commands/prediction_commands.py
"""
Command handlers for prediction workflows - Phase 3 Application Layer

Clean interface between CLI and domain logic with proper error handling.
"""

import sys
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from application.workflows.prediction_workflow import (
    GenerateCompletePredictionsWorkflow,
    PredictionWorkflowConfig,
)
from domain.services.edge_calculator import EdgeCalculator
from domain.services.ppi_calculator import PPICalculator
from infrastructure.data.repositories.csv_match_repository import (
    CSVFixtureRepository,
    CSVMatchRepository,
)
from shared.types.common_types import LeagueName, Season


class PredictCommand:
    """
    Handle prediction command with clean separation of concerns.

    Replaces complex logic in main.py with focused command handler.
    """

    def __init__(self):
        self.name = "predict"
        self.description = (
            "Generate enhanced predictions with PPI analysis and betting edges"
        )
        self.help_text = """
Generate comprehensive football predictions with:
- PPI (Point Performance Index) analysis
- Betting edge calculations  
- Recommendations with confidence scores

Usage:
  predict [OPTIONS]

Options:
  --league LEAGUE        Filter by specific league (e.g., Premier-League)
  --min-edge DECIMAL     Minimum edge for betting recommendations (default: 0.02)
  --max-predictions INT  Maximum predictions to generate (default: 50)
  --output-file PATH     Save results to CSV file
  --verbose              Show detailed progress information

Examples:
  predict                              # All leagues, default settings
  predict --league Premier-League      # Premier League only  
  predict --min-edge 0.03              # Higher edge threshold
  predict --verbose                    # Show detailed progress
        """

    def handle(self, args) -> int:
        """Handle prediction command with error handling and user feedback"""

        try:
            # Parse arguments
            config, league_filter, output_file, verbose = self._parse_args(args)

            if verbose:
                print("🚀 Enhanced Prediction System - Phase 3 Architecture")
                print("=" * 60)
                print(f"Configuration:")
                print(f"  Min betting edge: {config.min_betting_edge}")
                print(f"  Max PPI differential: {config.max_ppi_differential}")
                print(f"  Max predictions: {config.max_predictions}")
                if league_filter:
                    print(f"  League filter: {league_filter}")
                print()

            # Initialize repositories and services
            if verbose:
                print("🔧 Initializing domain services...")

            match_repo = CSVMatchRepository()
            fixture_repo = CSVFixtureRepository()
            ppi_calculator = PPICalculator()
            edge_calculator = EdgeCalculator(model_weight=config.model_weight)

            # Create workflow
            workflow = GenerateCompletePredictionsWorkflow(
                match_repository=match_repo,
                fixture_repository=fixture_repo,
                ppi_calculator=ppi_calculator,
                edge_calculator=edge_calculator,
                config=config,
            )

            # Execute workflow
            if verbose:
                print("⚡ Executing prediction workflow...\n")

            result = workflow.execute(league_filter)

            # Handle results
            if result.errors:
                print("❌ Errors occurred during prediction:")
                for error in result.errors:
                    print(f"  - {error}")
                print()

            if result.warnings:
                print("⚠️  Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
                print()

            # Display results
            self._display_results(result, verbose)

            # Save to file if requested
            if output_file:
                self._save_results(result, output_file)
                print(f"💾 Results saved to {output_file}")

            # Return appropriate exit code
            if result.errors:
                return 1  # Errors occurred
            elif not result.enhanced_predictions:
                print("ℹ️  No predictions generated")
                return 0
            else:
                return 0  # Success

        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
            return 130
        except Exception as e:
            print(f"❌ Prediction command failed: {e}")
            if "--verbose" in args:
                import traceback

                traceback.print_exc()
            return 1

    def _parse_args(self, args: List[str]):
        """Parse command line arguments"""
        config = PredictionWorkflowConfig()
        league_filter = None
        output_file = None
        verbose = False

        i = 0
        while i < len(args):
            arg = args[i]

            if arg == "--league":
                if i + 1 < len(args):
                    league_filter = LeagueName(args[i + 1])
                    i += 2
                else:
                    raise ValueError("--league requires a value")

            elif arg == "--min-edge":
                if i + 1 < len(args):
                    config.min_betting_edge = Decimal(args[i + 1])
                    i += 2
                else:
                    raise ValueError("--min-edge requires a value")

            elif arg == "--max-predictions":
                if i + 1 < len(args):
                    config.max_predictions = int(args[i + 1])
                    i += 2
                else:
                    raise ValueError("--max-predictions requires a value")

            elif arg == "--output-file":
                if i + 1 < len(args):
                    output_file = Path(args[i + 1])
                    i += 2
                else:
                    raise ValueError("--output-file requires a value")

            elif arg == "--verbose":
                verbose = True
                i += 1

            elif arg in ["--help", "-h"]:
                print(self.help_text)
                sys.exit(0)

            else:
                raise ValueError(f"Unknown argument: {arg}")

        return config, league_filter, output_file, verbose

    def _display_results(self, result, verbose: bool):
        """Display results in user-friendly format"""

        print("📊 PREDICTION RESULTS")
        print("=" * 50)

        # Summary stats
        stats = result.summary_stats
        print(f"Total Predictions: {stats.get('total_predictions', 0)}")
        print(f"Betting Candidates: {stats.get('betting_candidates', 0)}")
        print(f"Leagues Analyzed: {stats.get('leagues_analyzed', 0)}")

        if stats.get("predictions_with_ppi"):
            print(f"Predictions with PPI: {stats['predictions_with_ppi']}")
            print(f"Close Matches (PPI < 0.5): {stats.get('close_matches', 0)}")

        print(f"Execution Time: {result.execution_time:.2f}s")
        print()

        # Betting candidates
        if result.betting_candidates:
            print("🎯 BETTING CANDIDATES")
            print("-" * 30)

            for i, candidate in enumerate(result.betting_candidates[:10], 1):
                opp = candidate.betting_opportunity
                fixture = candidate.fixture

                print(f"{i:2d}. {fixture.home_team} vs {fixture.away_team}")
                print(f"    League: {fixture.league}")
                print(f"    Date: {fixture.date.strftime('%Y-%m-%d')}")
                print(
                    f"    Bet: {opp.bet_type.value} at {opp.recommended_odds.decimal}"
                )
                print(f"    Edge: {opp.edge:.3f} ({opp.edge * 100:.1f}%)")
                print(f"    Model Prob: {opp.model_probability.value:.3f}")

                if candidate.ppi_differential:
                    print(f"    PPI Diff: {candidate.ppi_differential:.3f}")

                print(f"    💡 {candidate.recommendation}")
                print()

            if len(result.betting_candidates) > 10:
                print(f"... and {len(result.betting_candidates) - 10} more candidates")
                print()

        # Top predictions (non-betting)
        non_betting = [
            p for p in result.enhanced_predictions if not p.betting_opportunity
        ][:5]

        if non_betting and verbose:
            print("📈 TOP ANALYSIS OPPORTUNITIES")
            print("-" * 35)

            for i, pred in enumerate(non_betting, 1):
                fixture = pred.fixture
                print(
                    f"{i}. {fixture.home_team} vs {fixture.away_team} ({fixture.league})"
                )

                if pred.ppi_differential:
                    print(f"   PPI Differential: {pred.ppi_differential:.3f}")

                if pred.edge_analysis:
                    print(
                        f"   Model Edge: {pred.edge_analysis.best_edge:.3f} on {pred.edge_analysis.best_bet_type.value}"
                    )

                print(f"   💡 {pred.recommendation}")
                print()

    def _save_results(self, result, output_file: Path):
        """Save results to CSV file"""
        import pandas as pd

        # Prepare data for CSV export
        csv_data = []

        for pred in result.enhanced_predictions:
            fixture = pred.fixture
            row = {
                "Date": fixture.date.strftime("%Y-%m-%d"),
                "League": fixture.league,
                "Home": fixture.home_team,
                "Away": fixture.away_team,
                "Recommendation": pred.recommendation,
                "Confidence_Score": float(pred.confidence_score),
            }

            # PPI data
            if pred.home_ppi:
                row.update(
                    {
                        "Home_PPI": float(pred.home_ppi.ppi_value),
                        "Home_PPG": float(pred.home_ppi.points_per_game),
                    }
                )

            if pred.away_ppi:
                row.update(
                    {
                        "Away_PPI": float(pred.away_ppi.ppi_value),
                        "Away_PPG": float(pred.away_ppi.points_per_game),
                    }
                )

            if pred.ppi_differential:
                row["PPI_Differential"] = float(pred.ppi_differential)

            # Betting data
            if pred.betting_opportunity:
                opp = pred.betting_opportunity
                row.update(
                    {
                        "Bet_Type": opp.bet_type.value,
                        "Betting_Edge": float(opp.edge),
                        "Recommended_Odds": float(opp.recommended_odds.decimal),
                        "Model_Probability": float(opp.model_probability.value),
                        "Fair_Odds": float(opp.fair_odds.decimal),
                        "Expected_Value": float(opp.expected_value),
                        "Is_Betting_Candidate": True,
                    }
                )
            else:
                row["Is_Betting_Candidate"] = False

            # Edge analysis
            if pred.edge_analysis:
                row.update(
                    {
                        "Best_Model_Edge": float(pred.edge_analysis.best_edge),
                        "Best_Model_Bet": pred.edge_analysis.best_bet_type.value,
                    }
                )

            csv_data.append(row)

        # Save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
