# Modified main.py with ZSD Poisson integration
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from config import DEFAULT_CONFIG, END_DATE, TODAY, AppConfig, Leagues
from processing import LeagueProcessor, get_historical_ppi
from utils.datetime_helpers import format_date
from utils.team_name_dict_builder import TeamNameManagerCLI

# Import ZSD integration
from zsd_integration import (
    daily_model_fitting,
    generate_zsd_predictions,
    periodic_parameter_optimization,
    setup_zsd_integration,
)


def build_team_name_dictionary():
    data_sources = ["fbref", "fbduk"]
    csv_path = "team_name_dictionary.csv"

    # Create manager
    manager = TeamNameManagerCLI(csv_path, data_sources)

    fbduk_teams = np.unique(
        pd.concat(
            [
                pd.read_csv(str(file))[["Home", "Away"]]
                for file in DEFAULT_CONFIG.fbduk_data_dir.rglob("*.csv")
                if file.is_file()
            ]
        )
        .to_numpy()
        .flatten()
    )

    fbref_teams = np.unique(
        pd.concat(
            [
                pd.read_csv(str(file))[["Home", "Away"]]
                for file in DEFAULT_CONFIG.fbref_data_dir.rglob("*.csv")
                if file.is_file()
            ]
        )
        .to_numpy()
        .flatten()
    )

    manager.import_team_list(
        fbref_teams, "fbref", auto_match=True, auto_threshold=0.7, interactive=True
    )

    manager.import_team_list(
        fbduk_teams, "fbduk", auto_match=True, auto_threshold=0.7, interactive=True
    )

    print(f"Dictionary saved to {csv_path}")


def merge_historical_odds_data():
    fbduk_odds_data = pd.concat(
        [
            pd.read_csv(str(file))
            for file in DEFAULT_CONFIG.fbduk_data_dir.rglob("*.csv")
            if file.is_file()
        ]
    )

    fbref_historical_rpi_data = pd.read_csv("historical_ppi.csv")

    print(f"fbduk input matches (odds): {fbduk_odds_data.shape[0]}")
    print(f"fbref imput matches: {fbref_historical_rpi_data.shape[0]}")

    team_name_dict = pd.read_csv("team_name_dictionary.csv")

    fbduk_to_fbref = dict(zip(team_name_dict["fbduk"], team_name_dict["fbref"]))

    def map_team_name(team_name):
        return fbduk_to_fbref.get(team_name, team_name)

    fbduk_odds_data["Home"] = fbduk_odds_data["Home"].apply(map_team_name)
    fbduk_odds_data["Away"] = fbduk_odds_data["Away"].apply(map_team_name)

    merged_df = (
        pd.merge(
            fbref_historical_rpi_data,
            fbduk_odds_data,
            on=["Date", "Home", "Away"],
            how="left",
        ).sort_values("Date")
    ).rename({"Season_x": "Season", "Wk_x": "Wk"}, axis="columns")

    print(f"Merged historical odds size: {merged_df.shape[0]}")
    print(f"{merged_df[merged_df['PSCH'].isna()].shape[0]} unmerged historical odds")

    merged_df.to_csv("historical_ppi_and_odds.csv", index=False)


def merge_future_odds_data():
    latest_ppi = pd.read_csv("latest_ppi.csv")
    fbduk_main_odds_data = pd.read_csv("fixtures.csv").rename(
        {"HomeTeam": "Home", "AwayTeam": "Away"}, axis="columns"
    )[
        [
            "Date",
            "Home",
            "Away",
            "PSH",
            "PSD",
            "PSA",
        ]
    ]

    # fbduk_extra_odds_data = pd.read_csv("new_league_fixtures.csv").rename(
    #     {"HomeTeam": "Home", "AwayTeam": "Away"}, axis="columns"
    # )[
    #     [
    #         "Date",
    #         "Home",
    #         "Away",
    #         "PSH",
    #         "PSD",
    #         "PSA",
    #     ]
    # ]

    # fbduk_odds_data = pd.concat([fbduk_main_odds_data, fbduk_extra_odds_data])
    fbduk_odds_data = fbduk_main_odds_data

    fbduk_odds_data = format_date(fbduk_odds_data)

    team_name_dict = pd.read_csv("team_name_dictionary.csv")

    fbduk_to_fbref = dict(zip(team_name_dict["fbduk"], team_name_dict["fbref"]))

    def map_team_name(team_name):
        return fbduk_to_fbref.get(team_name, team_name)

    fbduk_odds_data["Home"] = fbduk_odds_data["Home"].apply(map_team_name)
    fbduk_odds_data["Away"] = fbduk_odds_data["Away"].apply(map_team_name)

    merged_df = pd.merge(
        latest_ppi,
        fbduk_odds_data,
        on=["Date", "Home", "Away"],
        how="left",
    ).sort_values("Date")

    columns = [
        "Wk",
        "Date",
        "League",
        "Home",
        "Away",
        "aOppPPG",
        "hOppPPG",
        "aPPG",
        "hPPG",
        "hPPI",
        "aPPI",
        "PPI_Diff",
        "PSH",
        "PSD",
        "PSA",
    ]
    merged_df = merged_df[columns].sort_values(by="PPI_Diff")
    print(f"Merged future odds size: {merged_df.shape[0]}")

    merged_df.to_csv("fixtures_ppi_and_odds.csv", index=False)


def check_parameter_optimization_needed():
    """
    Check if parameter optimization should be run.
    Run this at the start of each season or monthly.
    """
    zsd_config_dir = Path("zsd_configs")

    # If no configs exist, optimization is needed
    if not zsd_config_dir.exists() or len(list(zsd_config_dir.glob("*.json"))) == 0:
        return True

    # Check if any config is older than 60 days
    from datetime import timedelta

    cutoff_date = datetime.now() - timedelta(days=60)

    for config_file in zsd_config_dir.glob("*.json"):
        if config_file.stat().st_mtime < cutoff_date.timestamp():
            return True

    return False


def run_parameter_optimization():
    """
    Run ZSD parameter optimization for all leagues.
    This is computationally expensive, so run periodically.
    """
    print("=" * 60)
    print("RUNNING ZSD PARAMETER OPTIMIZATION")
    print("=" * 60)

    try:
        # Setup ZSD integration
        zsd_processor = setup_zsd_integration(DEFAULT_CONFIG)

        # Run optimization
        periodic_parameter_optimization(zsd_processor)

        print("Parameter optimization completed successfully")
        return True

    except Exception as e:
        print(f"Error in parameter optimization: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_get_data(season: str):
    config = AppConfig(season)
    for league in Leagues:
        processor = LeagueProcessor(league, config)
        try:
            processor.get_fbref_data()
            processor.get_fbduk_data()
        except Exception as e:
            print(f"There was an error getting data for {league.name} {season} \r\n")
            print(e)
            continue


def main():
    """
    Enhanced main function with ZSD Poisson integration.
    """
    print("=" * 60)
    print("STARTING BETTING PIPELINE WITH ZSD INTEGRATION")
    print("=" * 60)

    # Step 1: Check if parameter optimization is needed
    if check_parameter_optimization_needed():
        print("Parameter optimization needed - running now...")
        if not run_parameter_optimization():
            print("Warning: Parameter optimization failed, continuing with defaults")
    else:
        print("Parameter optimization not needed (configs are recent)")

    # Step 2: Initialize ZSD processor
    zsd_processor = setup_zsd_integration(DEFAULT_CONFIG)

    # Step 3: Run your existing pipeline
    print("\n" + "=" * 60)
    print("RUNNING EXISTING PIPELINE (PPI + Basic ZSD)")
    print("=" * 60)

    ppi_all_leagues = []

    for league in Leagues:
        print(f"Processing {league.name} ({league.value['fbref_name']})")

        processor = LeagueProcessor(league, DEFAULT_CONFIG)

        ppi = processor.get_points_performance_index()

        if ppi:
            ppi_all_leagues.extend(ppi)

    # Step 4: Save your existing PPI results
    if ppi_all_leagues:
        print(f"Getting PPI betting candidates for the period {TODAY} to {END_DATE}")
        ppi_latest = pd.DataFrame(ppi_all_leagues).sort_values(by="PPI_Diff")

        # ppi_latest = ppi_latest[
        #     ppi_latest["PPI_Diff"] <= DEFAULT_CONFIG.ppi_diff_threshold
        # ]

        ppi_latest.to_csv("latest_ppi.csv", index=False)

    # Step 6: Process historical data
    print("\n" + "=" * 60)
    print("PROCESSING HISTORICAL DATA")
    print("=" * 60)

    historical_ppi = get_historical_ppi(DEFAULT_CONFIG)

    # Handle matches terminated during season more generally if needs be
    historical_ppi = historical_ppi[
        ~historical_ppi["Home"].eq("Reus") & ~historical_ppi["Away"].eq("Reus")
    ]

    historical_ppi.to_csv("historical_ppi.csv", index=False)
    merge_historical_odds_data()
    merge_future_odds_data()

    # Step 7: Run enhanced ZSD predictions
    print("\n" + "=" * 60)
    print("RUNNING ENHANCED ZSD PREDICTIONS")
    print("=" * 60)

    try:
        # Load the merged future data for ZSD predictions
        fixtures_with_odds = pd.read_csv("fixtures_ppi_and_odds.csv")

        # Fit ZSD models with latest historical data
        print("Fitting ZSD models with latest data...")
        daily_model_fitting(zsd_processor)

        # Generate enhanced ZSD predictions
        print("Generating enhanced ZSD predictions...")
        zsd_enhanced_predictions = generate_zsd_predictions(
            zsd_processor, fixtures_with_odds
        )

        if zsd_enhanced_predictions:
            # Save all predictions
            zsd_enhanced_df = pd.DataFrame(zsd_enhanced_predictions)
            zsd_enhanced_df.to_csv("latest_zsd_enhanced.csv", index=False)

            # Extract and display betting candidates
            betting_candidates = zsd_enhanced_df[
                zsd_enhanced_df["Is_Betting_Candidate"] == True
            ]

            if len(betting_candidates) > 0:
                print(
                    f"\nFound {len(betting_candidates)} enhanced ZSD betting candidates:"
                )
                print("-" * 80)

                # Sort by edge (highest first)
                betting_candidates = betting_candidates.sort_values(
                    "Edge", ascending=False
                )

                for idx, candidate in betting_candidates.head(15).iterrows():
                    print(
                        f"{candidate['Home']} vs {candidate['Away']} ({candidate['League']})"
                    )
                    print(f"  Date: {candidate['Date']}")
                    print(
                        f"  Recommended Bet: {candidate.get('Bet_Type', 'N/A')} at {candidate.get('Odds', 'N/A')} (Edge: {candidate.get('Edge', 0):.3f})"
                    )
                    print(
                        f"  ZSD Probabilities: H={candidate['ZSD_Prob_H']:.3f}, D={candidate['ZSD_Prob_D']:.3f}, A={candidate['ZSD_Prob_A']:.3f}"
                    )
                    if "PPI_Diff" in candidate:
                        print(f"  PPI_Diff: {candidate['PPI_Diff']:.3f}")
                    print()

                # Save betting candidates separately
                betting_candidates.to_csv("zsd_betting_candidates.csv", index=False)
                print(
                    f"Saved {len(betting_candidates)} betting candidates to zsd_betting_candidates.csv"
                )

            else:
                print(
                    "No enhanced ZSD betting candidates found with current thresholds"
                )

            # Summary statistics
            print(f"\nZSD PREDICTION SUMMARY:")
            print(f"Total predictions: {len(zsd_enhanced_df)}")
            print(f"Betting candidates: {len(betting_candidates)}")

            # Show distribution by league
            league_summary = (
                zsd_enhanced_df.groupby("League")
                .agg({"Is_Betting_Candidate": "sum", "ZSD_Prob_H": "count"})
                .rename(
                    columns={
                        "ZSD_Prob_H": "Total_Matches",
                        "Is_Betting_Candidate": "Betting_Candidates",
                    }
                )
            )

            print(f"\nBy League:")
            print(league_summary)

        else:
            print("No enhanced ZSD predictions generated")

    except FileNotFoundError as e:
        print(f"Could not run enhanced ZSD predictions: {e}")
        print(
            "Make sure historical_ppi_and_odds.csv and fixtures_ppi_and_odds.csv exist"
        )
    except Exception as e:
        print(f"Error in enhanced ZSD predictions: {e}")
        import traceback

        traceback.print_exc()

    # Step 8: Compare methods (optional)
    print("\n" + "=" * 60)
    print("COMPARING PREDICTION METHODS")
    print("=" * 60)

    try:
        # Load all prediction files
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
                    comparison_summary[method] = {
                        "total_predictions": len(df),
                        "betting_candidates": n_candidates,
                    }
                else:
                    comparison_summary[method] = {
                        "total_predictions": len(df),
                        "betting_candidates": len(
                            df
                        ),  # Assume all are candidates for PPI/basic ZSD
                    }

            except FileNotFoundError:
                comparison_summary[method] = {
                    "total_predictions": 0,
                    "betting_candidates": 0,
                }

        print("Method Comparison:")
        for method, stats in comparison_summary.items():
            print(
                f"  {method}: {stats['total_predictions']} predictions, {stats['betting_candidates']} candidates"
            )

    except Exception as e:
        print(f"Error in comparison: {e}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("Generated files:")
    print("  - latest_ppi.csv (PPI betting candidates)")
    print("  - latest_zsd.csv (Basic ZSD predictions)")
    print("  - latest_zsd_enhanced.csv (Enhanced ZSD predictions)")
    print("  - zsd_betting_candidates.csv (Top ZSD betting opportunities)")
    print("  - historical_ppi_and_odds.csv (Training data)")


def run_optimization_only():
    """
    Standalone function to run only parameter optimization.
    Use this when you want to update parameters without running the full pipeline.
    """
    print("Running ZSD parameter optimization only...")
    return run_parameter_optimization()


def run_predictions_only():
    """
    Standalone function to run only ZSD predictions (assuming parameters are optimized).
    Use this for daily prediction updates.
    """
    print("Running ZSD predictions only...")

    try:
        # Initialize ZSD processor
        zsd_processor = setup_zsd_integration(DEFAULT_CONFIG)

        # Load fixtures
        fixtures_with_odds = pd.read_csv("fixtures_ppi_and_odds.csv")

        # Fit models and generate predictions
        daily_model_fitting(zsd_processor)
        zsd_predictions = generate_zsd_predictions(zsd_processor, fixtures_with_odds)

        if zsd_predictions:
            zsd_df = pd.DataFrame(zsd_predictions)
            zsd_df.to_csv("latest_zsd_enhanced.csv", index=False)

            candidates = zsd_df[zsd_df["Is_Betting_Candidate"] == True]
            if len(candidates) > 0:
                candidates.to_csv("zsd_betting_candidates.csv", index=False)
                print(
                    f"Generated {len(zsd_predictions)} predictions with {len(candidates)} betting candidates"
                )
            else:
                print(
                    f"Generated {len(zsd_predictions)} predictions but no betting candidates"
                )
        else:
            print("No predictions generated")

    except Exception as e:
        print(f"Error running predictions: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "get_data":
            if len(sys.argv) > 2:  # season
                run_get_data(sys.argv[2])
        elif mode == "update_teams":
            build_team_name_dictionary()
        elif mode == "optimize":
            run_optimization_only()
        elif mode == "predict":
            run_predictions_only()
        elif mode == "full":
            main()
        else:
            print("Usage: python main.py [optimize|predict|full]")
            print("  optimize: Run parameter optimization only")
            print("  predict: Run predictions only")
            print("  full: Run full pipeline (default)")
            main()
    else:
        main()
