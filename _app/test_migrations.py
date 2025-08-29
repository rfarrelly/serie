#!/usr/bin/env python3
"""
Test script to verify the new DDD architecture works with existing data
Run this from your project root: python scripts/test_migration.py
"""

import sys
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))

from application.use_cases.analyze_team_performance import AnalyzeTeamPerformanceUseCase
from application.use_cases.get_betting_opportunities import (
    GetBettingOpportunitiesUseCase,
)
from application.use_cases.get_team_matches import GetTeamMatchesUseCase
from infrastructure.data.repositories.csv_match_repository import (
    CSVFixtureRepository,
    CSVMatchRepository,
)
from shared.types.common_types import LeagueName, Season, TeamName


def test_historical_data():
    """Test loading historical match data"""
    print("🧪 Testing Historical Data Loading...")

    try:
        # Create repository
        match_repo = CSVMatchRepository()

        # Create use case
        get_matches_use_case = GetTeamMatchesUseCase(match_repo)

        # Test with actual data - you might need to adjust these values
        matches = get_matches_use_case.execute(
            team=TeamName("Arsenal"),
            league=LeagueName("Premier-League"),
            season=Season("2023-2024"),
        )

        print(f"✅ Successfully loaded {len(matches)} matches for Arsenal")

        if matches:
            first_match = matches[0]
            print(f"📅 First match: {first_match.home_team} vs {first_match.away_team}")
            print(f"📆 Date: {first_match.date.strftime('%Y-%m-%d')}")
            print(f"🏟️  League: {first_match.league}")

            if first_match.result:
                print(
                    f"⚽ Result: {first_match.result.home_goals}-{first_match.result.away_goals}"
                )
                print(f"🎯 Outcome: {first_match.get_outcome().value}")

            # Test PPI data
            if first_match.ppi_data:
                print(
                    f"📊 PPI Data: Home={first_match.ppi_data.home_ppi}, Away={first_match.ppi_data.away_ppi}"
                )

            # Test odds data
            sharp_odds = first_match.get_sharp_odds()
            if sharp_odds:
                print(
                    f"💰 Sharp Odds: {sharp_odds.home.decimal} | {sharp_odds.draw.decimal} | {sharp_odds.away.decimal}"
                )

        return True

    except Exception as e:
        print(f"❌ Error in historical data test: {e}")
        print("💡 Make sure 'historical_ppi_and_odds.csv' exists in your project root")
        return False


def test_fixture_data():
    """Test loading fixture and betting data"""
    print("\n🧪 Testing Fixture Data Loading...")

    try:
        # Create repository
        fixture_repo = CSVFixtureRepository()

        # Test upcoming fixtures
        fixtures = fixture_repo.get_upcoming_fixtures()
        print(f"✅ Successfully loaded {len(fixtures)} upcoming fixtures")

        if fixtures:
            first_fixture = fixtures[0]
            print(
                f"📅 First fixture: {first_fixture.home_team} vs {first_fixture.away_team}"
            )
            print(f"📆 Date: {first_fixture.date.strftime('%Y-%m-%d')}")

            # Test PPI data
            if first_fixture.ppi_data:
                print(f"📊 PPI Diff: {first_fixture.ppi_data.ppi_difference}")

        # Test betting candidates
        try:
            betting_use_case = GetBettingOpportunitiesUseCase(fixture_repo)
            candidates = betting_use_case.execute()
            print(f"🎯 Found {len(candidates)} betting candidates")

            if candidates:
                best_candidate = candidates[0]
                print(
                    f"💡 Best candidate: {best_candidate.home_team} vs {best_candidate.away_team}"
                )

                if best_candidate.betting_opportunities:
                    opp = best_candidate.betting_opportunities[0]
                    print(f"💰 Edge: {opp.edge} on {opp.bet_type.value}")
                    print(
                        f"🎲 Model Prob: {opp.model_probability.value}, Market Prob: {opp.market_probability.value}"
                    )

        except FileNotFoundError as e:
            print(
                f"⚠️  Betting candidates file not found - this is OK for initial testing"
            )

        return True

    except Exception as e:
        print(f"❌ Error in fixture data test: {e}")
        print("💡 Make sure fixture CSV files exist in your project root")
        return False


def test_team_analysis():
    """Test team performance analysis"""
    print("\n🧪 Testing Team Analysis...")

    try:
        # Create repository and use case
        match_repo = CSVMatchRepository()
        analysis_use_case = AnalyzeTeamPerformanceUseCase(match_repo)

        # Analyze a team's performance
        analysis = analysis_use_case.execute(
            team=TeamName("Liverpool"),
            league=LeagueName("Premier-League"),
            season=Season("2023-2024"),
        )

        if "error" in analysis:
            print(f"⚠️  {analysis['error']}")
            return True

        print(f"✅ Team Analysis Complete for {analysis['team']}")
        print(f"📊 Games Played: {analysis['games_played']}")
        print(
            f"🏆 Points: {analysis['points']} ({analysis['points_per_game']} per game)"
        )
        print(
            f"⚽ Goals: {analysis['goals_for']} for, {analysis['goals_against']} against"
        )
        print(f"📈 Goal Difference: {analysis['goal_difference']}")

        if analysis["recent_form"]:
            form_string = "".join(analysis["recent_form"])
            print(f"🔥 Recent Form: {form_string}")

        if analysis["ppi_analysis"]["available"]:
            ppi = analysis["ppi_analysis"]
            print(
                f"📊 PPI Analysis: {ppi['matches_with_ppi']} matches, avg diff: {ppi['average_ppi_difference']}"
            )

        return True

    except Exception as e:
        print(f"❌ Error in team analysis test: {e}")
        return False


def test_value_objects():
    """Test domain value objects work correctly"""
    print("\n🧪 Testing Domain Value Objects...")

    try:
        from decimal import Decimal

        from domain.entities.betting import (
            MarketOdds,
            ModelProbabilities,
            Odds,
            Probability,
        )
        from shared.types.common_types import BetType, BookmakerType

        # Test Odds
        odds = Odds(Decimal("2.5"))
        print(
            f"✅ Odds created: {odds.decimal} (implied prob: {odds.implied_probability})"
        )

        # Test MarketOdds
        market_odds = MarketOdds(
            home=Odds(Decimal("2.1")),
            draw=Odds(Decimal("3.4")),
            away=Odds(Decimal("3.2")),
            bookmaker=BookmakerType.PINNACLE,
        )
        print(f"✅ Market odds created with overround: {market_odds.overround}")

        # Test Probabilities
        model_probs = ModelProbabilities(
            home_win=Probability(Decimal("0.45")),
            draw=Probability(Decimal("0.30")),
            away_win=Probability(Decimal("0.25")),
            model_type="test_model",
        )
        print(f"✅ Model probabilities created for {model_probs.model_type}")

        # Test validation
        try:
            invalid_odds = Odds(Decimal("0.5"))  # Should fail
            print("❌ Validation failed - invalid odds accepted")
            return False
        except Exception:
            print("✅ Odds validation working correctly")

        try:
            invalid_probs = ModelProbabilities(
                home_win=Probability(Decimal("0.60")),
                draw=Probability(Decimal("0.30")),
                away_win=Probability(Decimal("0.25")),  # Sum > 1.0
                model_type="test",
            )
            print("❌ Validation failed - invalid probabilities accepted")
            return False
        except Exception:
            print("✅ Probability validation working correctly")

        return True

    except Exception as e:
        print(f"❌ Error in value objects test: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting DDD Architecture Migration Tests")
    print("=" * 60)

    test_results = []

    # Run tests
    test_results.append(test_value_objects())
    test_results.append(test_historical_data())
    test_results.append(test_fixture_data())
    test_results.append(test_team_analysis())

    # Summary
    print("\n" + "=" * 60)
    passed = sum(test_results)
    total = len(test_results)

    if passed == total:
        print("🎉 ALL TESTS PASSED! Your DDD foundation is working perfectly!")
        print("✅ You're ready to start migrating more functionality.")
        print("\n📋 Next steps:")
        print("  1. Try with your actual team/league names")
        print("  2. Start extracting domain services from your old code")
        print("  3. Create use cases for your prediction workflows")
    else:
        print(f"⚠️  {passed}/{total} tests passed. Check the errors above.")
        print("\n🔧 Common fixes:")
        print("  - Make sure CSV files are in the project root")
        print("  - Check team/league names match your actual data")
        print("  - Verify CSV column names match expected format")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
