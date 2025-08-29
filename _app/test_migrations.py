#!/usr/bin/env python3
"""
Test script to verify the new architecture works with existing data
"""

from pathlib import Path

from application.use_cases.get_team_matches import GetTeamMatchesUseCase
from infrastructure.data.repositories.csv_match_repository import CSVMatchRepository
from shared.types.common_types import LeagueName, Season, TeamName


def test_basic_migration():
    """Test that we can load existing data through the new architecture"""

    # Use your existing data directory
    data_dir = Path("DATA")  # Adjust to your actual data path

    # Create repository (infrastructure layer)
    match_repo = CSVMatchRepository(data_dir)

    # Create use case (application layer)
    get_matches_use_case = GetTeamMatchesUseCase(match_repo)

    # Test with real data
    try:
        matches = get_matches_use_case.execute(
            team=TeamName("Arsenal"),
            league=LeagueName("Premier-League"),
            season=Season("2023-2024"),
        )

        print(f"✅ Successfully loaded {len(matches)} matches for Arsenal")

        if matches:
            first_match = matches[0]
            print(f"First match: {first_match.home_team} vs {first_match.away_team}")
            print(f"Date: {first_match.date}")

            if first_match.result:
                print(
                    f"Result: {first_match.result.home_goals}-{first_match.result.away_goals}"
                )
                print(f"Outcome: {first_match.get_outcome()}")

        print("🎉 Migration foundation is working!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check your data paths and file structure")


if __name__ == "__main__":
    test_basic_migration()
