# test_ddd_structure.py
"""
Test script to verify the DDD structure works correctly
"""

from pathlib import Path

import pandas as pd


def create_sample_data():
    """Create sample data for testing"""
    # Create sample historical matches
    sample_data = {
        "Date": ["2023-08-12", "2023-08-13", "2023-08-14", "2023-08-15", "2023-08-16"],
        "Home": ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Tottenham"],
        "Away": ["Tottenham", "Liverpool", "Arsenal", "Chelsea", "Manchester City"],
        "FTHG": [2, 1, 3, 2, 0],
        "FTAG": [1, 1, 1, 0, 2],
        "League": ["Premier-League"] * 5,
        "Season": ["2023-24"] * 5,
        "Wk": [1, 1, 2, 2, 3],
        "PSH": [1.8, 2.1, 1.6, 1.4, 2.5],
        "PSD": [3.5, 3.2, 4.0, 4.5, 3.1],
        "PSA": [4.2, 3.8, 5.5, 7.0, 2.8],
    }

    # Create data directory and sample file
    data_dir = Path("data/fbref")
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(sample_data)
    df.to_csv(data_dir / "Premier-League_2023-24.csv", index=False)

    # Create upcoming matches
    upcoming_data = {
        "Date": ["2023-08-20", "2023-08-21"],
        "Home": ["Arsenal", "Chelsea"],
        "Away": ["Newcastle", "Brighton"],
        "League": ["Premier-League"] * 2,
        "Season": ["2023-24"] * 2,
        "Wk": [4, 4],
        "PSH": [1.7, 2.0],
        "PSD": [3.6, 3.3],
        "PSA": [4.5, 3.9],
    }

    upcoming_df = pd.DataFrame(upcoming_data)
    upcoming_df.to_csv(data_dir / "unplayed_Premier-League_2023-24.csv", index=False)

    print("Sample data created successfully")
    return data_dir


def test_basic_functionality():
    """Test basic functionality with sample data"""
    # Create sample data
    data_dir = create_sample_data()

    try:
        # Import the main application
        from main import create_application

        print("Creating application...")
        app = create_application()

        print("Testing prediction generation...")
        predictions, opportunities = app.app_service.generate_predictions(
            league="Premier-League", method="zip"
        )

        print(f"Generated {len(predictions)} predictions")
        print(f"Found {len(opportunities)} betting opportunities")

        if predictions:
            print("\nFirst prediction:")
            pred = predictions[0]
            print(f"  {pred.home_team} vs {pred.away_team}")
            print(
                f"  Probabilities: H={pred.home_win_prob:.3f}, D={pred.draw_prob:.3f}, A={pred.away_win_prob:.3f}"
            )

        if opportunities:
            print("\nFirst opportunity:")
            opp = opportunities[0]
            print(
                f"  {opp.home_team} vs {opp.away_team}: {opp.bet_type} at {opp.odds:.2f}"
            )
            print(f"  Edge: {opp.edge:.3f}")

        return True

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def debug_data_loading():
    """Debug data loading issues"""
    from infrastructure.storage.csv_storage import CSVMatchRepository

    data_dir = Path("data/fbref")
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist")
        return

    print(f"Checking data directory: {data_dir}")
    csv_files = list(data_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f}")

    repo = CSVMatchRepository(data_dir)

    print("\nTesting played matches...")
    played_matches = repo.find_played_matches()
    print(f"Loaded {len(played_matches)} played matches")

    print("\nTesting upcoming matches...")
    upcoming_matches = repo.find_upcoming_matches()
    print(f"Loaded {len(upcoming_matches)} upcoming matches")

    if played_matches:
        print(
            f"\nFirst played match: {played_matches[0].home_team} vs {played_matches[0].away_team}"
        )
        print(f"  Goals: {played_matches[0].home_goals}-{played_matches[0].away_goals}")
        print(f"  Date: {played_matches[0].date}")

    if upcoming_matches:
        print(
            f"\nFirst upcoming match: {upcoming_matches[0].home_team} vs {upcoming_matches[0].away_team}"
        )
        print(f"  Date: {upcoming_matches[0].date}")


if __name__ == "__main__":
    print("Testing DDD Structure")
    print("=" * 50)

    # First test data loading
    debug_data_loading()

    print("\n" + "=" * 50)
    print("Testing full functionality")

    # Then test full functionality
    success = test_basic_functionality()

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
