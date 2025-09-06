from pathlib import Path
from typing import List

from application.services.application_service import ApplicationService
from shared.exceptions import DomainError, InfrastructureError

from ..formatters.output_formatter import OutputFormatter


class CLIApplication:
    def __init__(
        self, application_service: ApplicationService, output_formatter: OutputFormatter
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
                return self._handle_predictions(args[2:])
            elif command == "optimize":
                return self._handle_optimization(args[2:])
            elif command == "update":
                return self._handle_data_update(args[2:])
            elif command == "validate":
                return self._handle_validation(args[2:])
            elif command == "status":
                return self._handle_status()
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
            return 1

    def _handle_predictions(self, args: List[str]) -> int:
        # Parse arguments for prediction parameters
        league = None
        season = None
        method = "zip"

        # Simple argument parsing
        for i, arg in enumerate(args):
            if arg == "--league" and i + 1 < len(args):
                league = args[i + 1]
            elif arg == "--season" and i + 1 < len(args):
                season = args[i + 1]
            elif arg == "--method" and i + 1 < len(args):
                method = args[i + 1]

        print("Generating predictions...")
        predictions, betting_opportunities = self.app_service.generate_predictions(
            league=league, season=season, method=method
        )

        # Display results
        self.formatter.display_predictions(predictions)
        self.formatter.display_betting_opportunities(betting_opportunities)

        # Save results
        self.formatter.save_predictions_to_csv(predictions, "latest_predictions.csv")
        self.formatter.save_betting_opportunities_to_csv(
            betting_opportunities, "betting_opportunities.csv"
        )

        print(
            f"Generated {len(predictions)} predictions and found {len(betting_opportunities)} betting opportunities"
        )
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

        # Define leagues to update (this would come from config)
        leagues = [
            {"name": "Premier-League", "fbref_id": 9, "fbduk_id": "E0"},
            {"name": "Championship", "fbref_id": 10, "fbduk_id": "E1"},
            # Add more leagues as needed
        ]

        print(f"Updating data for season {season}...")
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
        print("System Status:")
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
            print(f"{status} {file}")

        return 0

    def _print_help(self):
        help_text = """
Football Betting Analysis Tool

Usage:
    python main.py <command> [options]

Commands:
    predict [--league LEAGUE] [--season SEASON] [--method METHOD]
        Generate predictions for upcoming matches
        
    optimize [LEAGUE]
        Optimize model parameters for all leagues or specific league
        
    update <season>
        Update data for specified season
        
    validate <betting_file> <predictions_file>
        Validate backtest results
        
    status
        Show system status
        
    help, -h, --help
        Show this help message

Examples:
    python main.py predict --league Premier-League --method zip
    python main.py optimize Premier-League
    python main.py update 2023-24
    python main.py validate results.csv predictions.csv
        """
        print(help_text)
