import sys
from pathlib import Path

import pandas as pd
from backtesting_validator import (
    analyze_market_efficiency_violations,
    benchmark_against_random_betting,
    manual_bet_inspection_helper,
    validate_zsd_backtest_results,
)
from config import DEFAULT_CONFIG, AppConfig, Leagues
from processing import LeagueProcessor, get_historical_ppi
from utils.data_merging import (
    merge_future_odds_data,
    merge_historical_odds_data,
)
from utils.pipeline_config import (
    PipelineConfig,
    PipelineModeValidator,
    print_pipeline_help,
)
from utils.team_name_dict_builder import build_team_name_dictionary
from zsd_integration import (
    daily_model_fitting,
    generate_zsd_predictions,
    periodic_parameter_optimization,
    setup_zsd_integration,
)


class BettingPipeline:
    """Main betting pipeline orchestrator."""

    def __init__(self, config=None):
        self.pipeline_config = PipelineConfig(config or DEFAULT_CONFIG)
        self.validator = PipelineModeValidator(self.pipeline_config)
        self.zsd_processor = None

    def _initialize_zsd(self):
        """Initialize ZSD processor if not already done."""
        if self.zsd_processor is None:
            self.zsd_processor = setup_zsd_integration(self.pipeline_config.base_config)
        return self.zsd_processor

    def run_zsd_predictions(self) -> bool:
        """Generate ZSD predictions with enhanced features."""
        if not self._validate_mode("predict"):
            return False

        print("=" * 60)
        print("RUNNING ENHANCED ZSD PREDICTIONS")
        print("=" * 60)

        try:
            # Load fixtures data
            fixtures_with_odds = pd.read_csv("fixtures_ppi_and_odds.csv")

            # Initialize ZSD processor and fit models
            zsd_processor = self._initialize_zsd()

            print("Fitting ZSD models with latest data...")
            daily_model_fitting(zsd_processor)

            print("Generating enhanced ZSD predictions...")
            zsd_predictions = generate_zsd_predictions(
                zsd_processor, fixtures_with_odds
            )

            if not zsd_predictions:
                print("No ZSD predictions generated")
                return False

            # Process and save predictions
            return self._process_zsd_results(zsd_predictions)

        except Exception as e:
            print(f"Error in ZSD predictions: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _process_zsd_results(self, zsd_predictions) -> bool:
        """Process and save ZSD prediction results."""
        zsd_df = pd.DataFrame(zsd_predictions)
        zsd_df.to_csv("latest_zsd_enhanced.csv", index=False)

        # Extract betting candidates
        betting_candidates = zsd_df[zsd_df["Is_Betting_Candidate"] == True]

        # if len(betting_candidates) > 0:
        self._display_betting_candidates(betting_candidates)
        betting_candidates.to_csv("zsd_betting_candidates.csv", index=False)
        print(f"Saved {len(betting_candidates)} betting candidates")
        # else:
        #     print("No enhanced ZSD betting candidates found with current thresholds")

        # Display summary
        self._display_prediction_summary(zsd_df, betting_candidates)
        return True

    def _validate_mode(self, mode: str) -> bool:
        """Validate mode and show any issues."""
        is_valid, errors, warnings = self.validator.validate_mode(mode)

        if warnings:
            print("Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
            print()

        if errors:
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
            print("Cannot proceed with current mode. Please fix the issues above.")
            return False

        return True

    def run_parameter_optimization(self) -> bool:
        """Run ZSD parameter optimization for all leagues."""
        if not self._validate_mode("optimize"):
            return False

        print("=" * 60)
        print("RUNNING ZSD PARAMETER OPTIMIZATION")
        print("=" * 60)

        try:
            zsd_processor = self._initialize_zsd()
            periodic_parameter_optimization(zsd_processor)
            print("Parameter optimization completed successfully")
            return True

        except Exception as e:
            print(f"Error in parameter optimization: {e}")
            import traceback

            traceback.print_exc()
            return False

    def run_get_data(self, season: str) -> bool:
        """Download data for all leagues for a specific season."""
        print(f"Downloading data for season: {season}")

        config = AppConfig(season)
        failed_leagues = []
        successful_leagues = []

        for league in Leagues:
            processor = LeagueProcessor(league, config)
            try:
                print(f"Processing {league.name}...")
                processor.get_fbref_data()
                processor.get_fbduk_data()
                successful_leagues.append(league.name)
            except Exception as e:
                print(f"Error getting data for {league.name} {season}: {e}\r\n")
                failed_leagues.append(league.name)
                continue

        # Summary
        print(f"\nData download summary:")
        print(f"  Successful: {len(successful_leagues)} leagues")
        print(f"  Failed: {len(failed_leagues)} leagues")

        if failed_leagues:
            print(f"  Failed leagues: {', '.join(failed_leagues)}")

        return len(failed_leagues) == 0

    def run_latest_ppi(self) -> bool:
        """Generate latest PPI data and merge with odds."""
        if not self._validate_mode("latest_ppi"):
            return False

        print("=" * 60)
        print("GENERATING LATEST PPI PREDICTIONS")
        print("=" * 60)

        ppi_all_leagues = []
        failed_leagues = []

        for league in Leagues:
            print(f"Processing {league.name} ({league.value['fbref_name']})")

            processor = LeagueProcessor(league, self.pipeline_config.base_config)

            try:
                ppi = processor.get_points_performance_index()
                if ppi:
                    ppi_all_leagues.extend(ppi)
            except Exception as e:
                print(f"Error getting latest PPI for {processor.league_name}: {e}")
                failed_leagues.append(league.name)
                continue

        if not ppi_all_leagues:
            print("No PPI data generated")
            return False

        # Save PPI data
        from config import END_DATE, TODAY

        print(f"Getting PPI betting candidates for the period {TODAY} to {END_DATE}")

        ppi_latest = pd.DataFrame(ppi_all_leagues).sort_values(by="PPI_Diff")
        ppi_latest.to_csv("latest_ppi.csv", index=False)

        # Merge with odds data
        try:
            merge_future_odds_data()
            print(f"Successfully generated {len(ppi_latest)} PPI predictions")
        except Exception as e:
            print(f"Error merging with odds data: {e}")
            return False

        if failed_leagues:
            print(f"Note: Failed leagues: {', '.join(failed_leagues)}")

        return True

    def run_historical_ppi(self) -> bool:
        """Generate historical PPI data and merge with odds."""
        if not self._validate_mode("historical_ppi"):
            return False

        print("=" * 60)
        print("GENERATING HISTORICAL PPI DATA")
        print("=" * 60)

        try:
            historical_ppi = get_historical_ppi(self.pipeline_config.base_config)

            # Handle matches terminated during season more generally if needs be
            historical_ppi = historical_ppi[
                ~historical_ppi["Home"].eq("Reus") & ~historical_ppi["Away"].eq("Reus")
            ]

            historical_ppi.to_csv("historical_ppi.csv", index=False)

            # Merge with odds data
            merge_historical_odds_data()
            print(
                f"Successfully generated {len(historical_ppi)} historical PPI records"
            )
            return True

        except Exception as e:
            print(f"Error generating historical PPI: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _display_betting_candidates(self, betting_candidates):
        """Display betting candidates in a formatted way with enhanced information."""
        print(f"\nFound {len(betting_candidates)} enhanced ZSD betting candidates:")
        print("-" * 100)

        # Sort by edge (highest first)
        betting_candidates = betting_candidates.sort_values("Edge", ascending=False)

        max_display = min(
            len(betting_candidates), self.pipeline_config.max_recommendations
        )

        for idx, candidate in betting_candidates.head(max_display).iterrows():
            print(f"{candidate['Home']} vs {candidate['Away']} ({candidate['League']})")
            print(f"  Date: {candidate['Date']}")
            odds = candidate.get("Odds", None)
            edge = candidate.get("Edge", 0)

            try:
                odds_str = f"{float(odds):.2f}"
            except (ValueError, TypeError):
                odds_str = "N/A"

            print(
                f"  Recommended Bet: {candidate.get('Bet_Type', 'N/A')} at "
                f"{odds_str} (Edge: {float(edge):.3f})"
            )

            # Display all prediction methods
            print(f"  Model Predictions:")
            print(
                f"    Poisson:  H={candidate.get('Poisson_Prob_H', 0):.3f}, D={candidate.get('Poisson_Prob_D', 0):.3f}, A={candidate.get('Poisson_Prob_A', 0):.3f}"
            )
            print(
                f"    ZIP:      H={candidate.get('ZIP_Prob_H', 0):.3f}, D={candidate.get('ZIP_Prob_D', 0):.3f}, A={candidate.get('ZIP_Prob_A', 0):.3f}"
            )
            print(
                f"    MOV:      H={candidate.get('MOV_Prob_H', 0):.3f}, D={candidate.get('MOV_Prob_D', 0):.3f}, A={candidate.get('MOV_Prob_A', 0):.3f}"
            )

            # Display combined probabilities and odds
            print(f"  Market Analysis:")
            print(
                f"    No-Vig Probs: H={candidate.get('NoVig_Prob_H', 0):.3f}, D={candidate.get('NoVig_Prob_D', 0):.3f}, A={candidate.get('NoVig_Prob_A', 0):.3f}"
            )
            print(
                f"    Model Avg:    H={candidate.get('ModelAvg_Prob_H', 0):.3f}, D={candidate.get('ModelAvg_Prob_D', 0):.3f}, A={candidate.get('ModelAvg_Prob_A', 0):.3f}"
            )
            print(
                f"    Weighted:     H={candidate.get('Weighted_Prob_H', 0):.3f}, D={candidate.get('Weighted_Prob_D', 0):.3f}, A={candidate.get('Weighted_Prob_A', 0):.3f}"
            )

            # Display fair odds
            print(
                f"  Fair Odds:    H={candidate.get('Fair_Odds_H', 0):.2f}, D={candidate.get('Fair_Odds_D', 0):.2f}, A={candidate.get('Fair_Odds_A', 0):.2f}"
            )
            print(
                f"  Market Odds:  H={candidate.get('PSH', 0):.2f}, D={candidate.get('PSD', 0):.2f}, A={candidate.get('PSA', 0):.2f}"
            )

            if "PPI_Diff" in candidate:
                print(f"  PPI_Diff: {candidate['PPI_Diff']:.3f}")
            print()

        if len(betting_candidates) > max_display:
            print(f"... and {len(betting_candidates) - max_display} more candidates")

    def _display_prediction_summary(self, zsd_df, betting_candidates):
        """Display prediction summary statistics."""
        print(f"\nZSD PREDICTION SUMMARY:")
        print(f"Total predictions: {len(zsd_df)}")
        print(f"Betting candidates: {len(betting_candidates)}")

        if len(betting_candidates) > 0:
            avg_edge = betting_candidates["Edge"].mean()
            max_edge = betting_candidates["Edge"].max()
            print(f"Average edge: {avg_edge:.3f}")
            print(f"Maximum edge: {max_edge:.3f}")

            # Show distribution of bet types
            bet_type_dist = betting_candidates["Bet_Type"].value_counts()
            print(f"Bet type distribution: {bet_type_dist.to_dict()}")

        # Show distribution by league
        league_summary = (
            zsd_df.groupby("League")
            .agg({"Is_Betting_Candidate": "sum", "ZIP_Prob_H": "count"})
            .rename(
                columns={
                    "ZIP_Prob_H": "Total_Matches",
                    "Is_Betting_Candidate": "Betting_Candidates",
                }
            )
        )

        print(f"\nBy League:")
        print(league_summary)

    def compare_prediction_methods(self):
        """Compare different prediction methods."""
        print("=" * 60)
        print("COMPARING PREDICTION METHODS")
        print("=" * 60)

        files_to_compare = {
            "PPI": "latest_ppi.csv",
            "Basic ZSD": "latest_zsd.csv",
            "Enhanced ZSD": "latest_zsd_enhanced.csv",
        }

        comparison_summary = {}

        for method, filename in files_to_compare.items():
            try:
                df = pd.read_csv(filename)

                if method == "Enhanced ZSD":
                    n_candidates = len(df[df["Is_Betting_Candidate"] == True])
                    # Also show prediction method breakdown if enhanced
                    if len(df) > 0 and "Poisson_Prob_H" in df.columns:
                        avg_poisson_home = df["Poisson_Prob_H"].mean()
                        avg_zip_home = df["ZIP_Prob_H"].mean()
                        avg_mov_home = df.get("MOV_Prob_H", pd.Series([0])).mean()
                        comparison_summary[method] = {
                            "total_predictions": len(df),
                            "betting_candidates": n_candidates,
                            "avg_poisson_home": avg_poisson_home,
                            "avg_zip_home": avg_zip_home,
                            "avg_mov_home": avg_mov_home,
                        }
                    else:
                        comparison_summary[method] = {
                            "total_predictions": len(df),
                            "betting_candidates": n_candidates,
                        }
                else:
                    # Assume all are candidates for PPI/basic ZSD
                    n_candidates = len(df)
                    comparison_summary[method] = {
                        "total_predictions": len(df),
                        "betting_candidates": n_candidates,
                    }

            except FileNotFoundError:
                comparison_summary[method] = {
                    "total_predictions": 0,
                    "betting_candidates": 0,
                }

        print("Method Comparison:")
        for method, stats in comparison_summary.items():
            print(
                f"  {method}: {stats['total_predictions']} predictions, "
                f"{stats['betting_candidates']} candidates"
            )
            if "avg_poisson_home" in stats:
                print(
                    f"    Avg Home Win Probs - Poisson: {stats['avg_poisson_home']:.3f}, ZIP: {stats['avg_zip_home']:.3f}, MOV: {stats['avg_mov_home']:.3f}"
                )

        return comparison_summary

    def run_backtest_validation(self, betting_csv, predictions_csv):
        print("Running comprehensive backtest validation...")
        # 1. Main validation
        validation_result = validate_zsd_backtest_results(
            betting_csv, predictions_csv, self.pipeline_config
        )

        if validation_result:
            print(f"\nValidation complete!")
            print(f"Valid: {validation_result.is_valid}")
            print(f"Confidence: {validation_result.confidence_score:.2f}")

        # 2. Manual inspection helper
        try:
            print(f"\nCreating manual inspection helper...")
            manual_bet_inspection_helper(betting_csv, n_samples=50)
        except FileNotFoundError:
            print("Betting results file not found - run backtest first")

        # 3. Additional validation checks
        print(f"\n" + "=" * 60)
        print("ADDITIONAL VALIDATION CHECKS")
        print("=" * 60)

        # Market efficiency check
        analyze_market_efficiency_violations(betting_csv)

        # Random betting benchmark
        benchmark_against_random_betting(betting_csv)

        # Cross-validation note
        print(f"\n" + "=" * 60)
        print("RECOMMENDED ADDITIONAL CHECKS")
        print("=" * 60)
        print("1. Run backtest on different time periods")
        print("2. Test on different leagues separately")
        print("3. Use walk-forward validation")
        print("4. Check results against betting exchange data")
        print("5. Paper trade for a few weeks before going live")

    def run_full_pipeline(self) -> bool:
        """Run the complete betting pipeline."""
        print("=" * 60)
        print("STARTING BETTING PIPELINE WITH ZSD INTEGRATION")
        print("=" * 60)

        # Show initial status
        self.pipeline_config.print_status_report()

        success_steps = 0
        total_steps = 5

        # Step 1: Check if parameter optimization is needed
        print("\n" + "=" * 60)
        print("STEP 1: PARAMETER OPTIMIZATION CHECK")
        print("=" * 60)

        if self.pipeline_config.check_zsd_optimization_needed():
            print("Parameter optimization needed - running now...")
            if self.run_parameter_optimization():
                success_steps += 1
                print("✓ Parameter optimization completed")
            else:
                print("✗ Parameter optimization failed, continuing with defaults")
        else:
            print("✓ Parameter optimization not needed (configs are recent)")
            success_steps += 1

        # Step 2: Generate latest PPI predictions
        print("\n" + "=" * 60)
        print("STEP 2: LATEST PPI PREDICTIONS")
        print("=" * 60)

        if self.run_latest_ppi():
            success_steps += 1
            print("✓ Latest PPI predictions completed")
        else:
            print("✗ Latest PPI predictions failed")
            return False  # Can't continue without PPI data

        # Step 3: Run ZSD predictions
        print("\n" + "=" * 60)
        print("STEP 3: ZSD PREDICTIONS")
        print("=" * 60)

        if self.run_zsd_predictions():
            success_steps += 1
            print("✓ ZSD predictions completed")
        else:
            print("✗ ZSD predictions failed")

        # Step 4: Compare methods
        print("\n" + "=" * 60)
        print("STEP 4: METHOD COMPARISON")
        print("=" * 60)

        try:
            self.compare_prediction_methods()
            success_steps += 1
            print("✓ Method comparison completed")
        except Exception as e:
            print(f"✗ Method comparison failed: {e}")

        # Step 5: Final summary
        print("\n" + "=" * 60)
        print("STEP 5: PIPELINE SUMMARY")
        print("=" * 60)

        success_rate = success_steps / total_steps * 100
        print(
            f"Pipeline completed: {success_steps}/{total_steps} steps successful ({success_rate:.0f}%)"
        )

        # List generated files
        output_files = []
        for name, path in self.pipeline_config.output_files.items():
            if Path(path).exists():
                file_size = Path(path).stat().st_size / 1024  # KB
                output_files.append(f"{path} ({file_size:.1f}KB)")

        if output_files:
            print(f"\nGenerated files:")
            for file_info in output_files:
                print(f"  - {file_info}")

        success_steps += 1
        return success_steps >= 4  # Consider successful if most steps completed


def main():
    """Main entry point for the betting pipeline."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print_pipeline_help()
        return

    pipeline = BettingPipeline()

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "status":
            pipeline.pipeline_config.print_status_report()

        elif mode == "get_data":
            if len(sys.argv) > 2:
                season = sys.argv[2]
                pipeline.run_get_data(season)
            else:
                print("Usage: python main.py get_data <season>")
                print("Example: python main.py get_data 2023-24")

        elif mode == "latest_ppi":
            pipeline.run_latest_ppi()

        elif mode == "historical_ppi":
            pipeline.run_historical_ppi()

        elif mode == "update_teams":
            build_team_name_dictionary()

        elif mode == "optimize":
            pipeline.run_parameter_optimization()

        elif mode == "validate":
            if len(sys.argv) > 2:
                betting_filename = sys.argv[2]
                prediction_filename = sys.argv[3]
                pipeline.run_backtest_validation(betting_filename, prediction_filename)
            else:
                print("Usage: uv run app/main.py validate <filename>")
                print(
                    "Example: uv run app/main.py validate optimisation_validation/betting_results/Premier-League_best_betting_results.csv "
                    "optimisation_validation/prediction_results/Premier-League_best_predictions.csv"
                )

        elif mode == "predict":
            pipeline.run_zsd_predictions()

        elif mode == "full":
            pipeline.run_full_pipeline()

        elif mode == "compare":
            pipeline.compare_prediction_methods()

        else:
            print(f"Unknown mode: {mode}")
            print_pipeline_help()

    else:
        # Default to full pipeline
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
