import sys
from datetime import datetime
from pathlib import Path
from typing import List

from application.services.application_service import ApplicationService
from shared.exceptions import DomainError, InfrastructureError

from ..formatters.output_formatter import OutputFormatter


class EnhancedCLIApplication:
    def __init__(
        self,
        application_service: ApplicationService,
        output_formatter: OutputFormatter,
    ):
        self.app_service = application_service
        self.formatter = output_formatter

    def run(self, args: List[str]) -> int:
        if len(args) < 2:
            self._print_help()
            return 1

        command = args[1].lower()

        try:
            if command == "predict":
                return self._handle_enhanced_predictions(args[2:])
            elif command == "optimize":
                return self._handle_optimization(args[2:])
            elif command == "update":
                return self._handle_data_update(args[2:])
            elif command == "validate":
                return self._handle_validation(args[2:])
            elif command == "status":
                return self._handle_status()
            elif command == "legacy":
                return self._handle_legacy_compatibility()
            elif command in ["-h", "--help", "help"]:
                self._print_help()
                return 0
            else:
                print(f"Unknown command: {command}")
                self._print_help()
                return 1

        except DomainError as e:
            print(f"Domain error: {e}")
            return 1
        except InfrastructureError as e:
            print(f"Infrastructure error: {e}")
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            return 1

    def _handle_enhanced_predictions(self, args: List[str]) -> int:
        # Parse arguments
        league = None
        season = None

        for i, arg in enumerate(args):
            if arg == "--league" and i + 1 < len(args):
                league = args[i + 1]
            elif arg == "--season" and i + 1 < len(args):
                season = args[i + 1]

        print("Generating enhanced predictions with all methods...")
        print(f"League: {league or 'All'}")
        print(f"Season: {season or 'Current'}")

        predictions, betting_opportunities = (
            self.app_service.generate_enhanced_predictions(league=league, season=season)
        )

        # Display results with all original detail
        self.formatter.display_enhanced_predictions(predictions)
        self.formatter.display_enhanced_betting_opportunities(betting_opportunities)

        # Save results in original format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_filename = f"latest_zsd_enhanced_{timestamp}.csv"
        opp_filename = f"zsd_betting_candidates_{timestamp}.csv"

        self.formatter.save_enhanced_predictions_to_csv(predictions, pred_filename)
        self.formatter.save_enhanced_betting_opportunities_to_csv(
            betting_opportunities, opp_filename
        )

        # Legacy filenames for compatibility
        self.formatter.save_enhanced_predictions_to_csv(
            predictions, "latest_zsd_enhanced.csv"
        )
        self.formatter.save_enhanced_betting_opportunities_to_csv(
            betting_opportunities, "zsd_betting_candidates.csv"
        )

        print(f"\nSUMMARY:")
        print(f"Total predictions: {len(predictions)}")
        print(f"Betting candidates: {len(betting_opportunities)}")

        if betting_opportunities:
            avg_edge = sum(opp.edge for opp in betting_opportunities) / len(
                betting_opportunities
            )
            max_edge = max(opp.edge for opp in betting_opportunities)
            print(f"Average edge: {avg_edge:.3f}")
            print(f"Maximum edge: {max_edge:.3f}")

            bet_type_dist = {}
            for opp in betting_opportunities:
                bet_type_dist[opp.bet_type] = bet_type_dist.get(opp.bet_type, 0) + 1
            print(f"Bet type distribution: {bet_type_dist}")

        return 0

    def _handle_optimization(self, args: List[str]) -> int:
        league = None
        if len(args) > 0:
            league = args[0]

        print(f"Optimizing parameters{' for ' + league if league else ''}...")
        results = self.app_service.optimize_model_parameters(league)

        self.formatter.display_optimization_results(results)
        return 0

    def _handle_data_update(self, args: List[str]) -> int:
        if len(args) < 1:
            print("Usage: update <season>")
            return 1

        season = args[0]

        # Define leagues to update (original league set)
        from shared.types import League

        leagues = []
        for league in League:
            leagues.append(
                {
                    "name": league.name,
                    "fbref_id": league.fbref_id,
                    "fbduk_id": league.fbduk_id,
                }
            )

        print(f"Updating data for season {season}...")
        print(f"Processing {len(leagues)} leagues...")

        results = self.app_service.update_league_data(leagues, season)

        self.formatter.display_update_results(results)
        return 0

    def _handle_validation(self, args: List[str]) -> int:
        if len(args) < 2:
            print("Usage: validate <betting_results_file> <predictions_file>")
            return 1

        betting_file = args[0]
        predictions_file = args[1]

        if not Path(betting_file).exists():
            print(f"Betting results file not found: {betting_file}")
            return 1

        if not Path(predictions_file).exists():
            print(f"Predictions file not found: {predictions_file}")
            return 1

        print("Validating backtest results...")
        validation_results = self.app_service.validate_backtest_results(
            betting_file, predictions_file
        )

        self.formatter.display_validation_results(validation_results)
        return 0

    def _handle_status(self) -> int:
        print("Enhanced System Status:")
        print("=" * 50)

        # Check file existence
        required_files = [
            "historical_ppi_and_odds.csv",
            "fixtures_ppi_and_odds.csv",
            "team_name_dictionary.csv",
        ]

        for file in required_files:
            exists = Path(file).exists()
            status = "✓" if exists else "✗"
            size = ""
            if exists:
                size_mb = Path(file).stat().st_size / (1024 * 1024)
                size = f" ({size_mb:.1f}MB)"
            print(f"{status} {file}{size}")

        # Check data directories
        data_dirs = ["data/fbref", "data/fbduk", "config", "output"]
        print(f"\nData Directories:")
        for dir_path in data_dirs:
            exists = Path(dir_path).exists()
            status = "✓" if exists else "✗"
            count = ""
            if exists and Path(dir_path).is_dir():
                csv_count = len(list(Path(dir_path).rglob("*.csv")))
                count = f" ({csv_count} CSV files)"
            print(f"{status} {dir_path}{count}")

        return 0

    def _handle_legacy_compatibility(self) -> int:
        """Handle legacy function calls for backward compatibility"""
        print("Legacy Compatibility Mode")
        print("=" * 40)
        print("Available legacy functions:")
        print("  - run_backtest_example()")
        print("  - run_calibration_example()")
        print("  - build_team_name_dictionary()")
        print("  - validate_zsd_backtest_results()")
        print("\nTo use legacy functions, import from legacy_main.py")
        print("New recommended usage: python enhanced_main.py predict")
        return 0

    def _print_help(self):
        help_text = """
Enhanced Football Betting Analysis Tool - Full Feature Set

Usage:
    python enhanced_main.py <command> [options]

Commands:
    predict [--league LEAGUE] [--season SEASON]
        Generate enhanced predictions with all methods (Poisson, ZIP, MOV)
        Includes PPI analysis, betting edge calculations, and comprehensive output
        
    optimize [LEAGUE]
        Optimize model parameters for all leagues or specific league
        
    update <season>
        Update data for all leagues for specified season
        
    validate <betting_file> <predictions_file>
        Comprehensive backtest validation with statistical tests
        
    status
        Show enhanced system status with data directory info
        
    legacy
        Show legacy compatibility information
        
    help, -h, --help
        Show this help message

Features (restored from original system):
    ✓ Multiple prediction methods (Poisson, ZIP, MOV)
    ✓ PPI (Points Performance Index) calculations
    ✓ Comprehensive betting edge analysis
    ✓ Kelly criterion stake sizing
    ✓ All probability and edge calculations
    ✓ Market efficiency analysis
    ✓ Team rating optimization
    ✓ Zero-inflated Poisson modeling
    ✓ Margin of Victory predictions
    ✓ Cross-validation and parameter tuning
    ✓ Backtest validation with statistical tests

Examples:
    python enhanced_main.py predict --league Premier-League
    python enhanced_main.py predict  # All leagues
    python enhanced_main.py optimize Premier-League
    python enhanced_main.py update 2023-24
    python enhanced_main.py validate results.csv predictions.csv
        """
        print(help_text)


def main():
    """Enhanced main entry point with full feature set"""
    try:
        app = create_enhanced_application()
        exit_code = app.run(sys.argv)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
