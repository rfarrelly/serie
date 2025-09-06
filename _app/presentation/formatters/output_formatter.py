from typing import Any, Dict, List

import pandas as pd
from application.dto.betting_dto import BettingOpportunityDTO
from application.dto.prediction_dto import PredictionResponse


class OutputFormatter:
    def __init__(self, max_display_items: int = 20):
        self.max_display_items = max_display_items

    def display_predictions(self, predictions: List[PredictionResponse]) -> None:
        if not predictions:
            print("No predictions available")
            return

        print(f"\n{len(predictions)} MATCH PREDICTIONS")
        print("=" * 60)

        display_count = min(len(predictions), self.max_display_items)

        for pred in predictions[:display_count]:
            print(f"\n{pred.home_team} vs {pred.away_team} ({pred.league})")
            print(f"Date: {pred.date}")
            print(
                f"Probabilities: H={pred.home_win_prob:.3f}, D={pred.draw_prob:.3f}, A={pred.away_win_prob:.3f}"
            )
            print(
                f"Expected Goals: {pred.expected_home_goals:.2f} - {pred.expected_away_goals:.2f}"
            )
            print(f"Model: {pred.model_type}")

        if len(predictions) > display_count:
            print(f"\n... and {len(predictions) - display_count} more predictions")

    def display_betting_opportunities(
        self, opportunities: List[BettingOpportunityDTO]
    ) -> None:
        if not opportunities:
            print("\nNo betting opportunities found")
            return

        print(f"\n{len(opportunities)} BETTING OPPORTUNITIES")
        print("=" * 60)

        # Sort by edge (highest first)
        sorted_opps = sorted(opportunities, key=lambda x: x.edge, reverse=True)
        display_count = min(len(sorted_opps), self.max_display_items)

        for opp in sorted_opps[:display_count]:
            print(f"\n{opp.home_team} vs {opp.away_team} ({opp.league})")
            print(f"Date: {opp.date}")
            print(f"Bet: {opp.bet_type} at {opp.odds:.2f}")
            print(f"Edge: {opp.edge:.3f} ({opp.edge * 100:.1f}%)")
            print(
                f"Model Prob: {opp.model_probability:.3f}, Market Prob: {opp.market_probability:.3f}"
            )
            print(f"Expected Value: ${opp.expected_value:.2f}")

        if len(opportunities) > display_count:
            print(f"\n... and {len(opportunities) - display_count} more opportunities")

    def display_optimization_results(self, results: Dict[str, Any]) -> None:
        print("\nOPTIMIZATION RESULTS")
        print("=" * 40)

        if "optimized_parameters" in results:
            params = results["optimized_parameters"]
            print(f"Decay Rate: {params.get('decay_rate', 'N/A')}")
            print(f"L2 Regularization: {params.get('l2_reg', 'N/A')}")
            print(f"Home Advantage: {params.get('home_advantage', 'N/A')}")
            print(f"Away Adjustment: {params.get('away_adjustment', 'N/A')}")

        print(f"Teams: {results.get('num_teams', 'N/A')}")
        print(f"Matches: {results.get('num_matches', 'N/A')}")

    def display_update_results(self, results: Dict[str, bool]) -> None:
        print("\nDATA UPDATE RESULTS")
        print("=" * 40)

        successful = sum(results.values())
        total = len(results)

        print(f"Successful: {successful}/{total}")

        for league, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {league}")

    def display_validation_results(self, results: Dict[str, Any]) -> None:
        print("\nBACKTEST VALIDATION RESULTS")
        print("=" * 50)

        print(f"Valid: {'✓' if results['is_valid'] else '✗'}")
        print(f"Confidence Score: {results['confidence_score']:.2f}/1.0")

        if results["red_flags"]:
            print(f"\n🚨 RED FLAGS ({len(results['red_flags'])}):")
            for flag in results["red_flags"]:
                print(f"  - {flag}")

        if results["warnings"]:
            print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"  - {warning}")

        # Display key metrics
        metrics = results.get("metrics", {})
        if metrics:
            print(f"\nKEY METRICS:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    def save_predictions_to_csv(
        self, predictions: List[PredictionResponse], filename: str
    ) -> None:
        if not predictions:
            return

        data = []
        for pred in predictions:
            data.append(
                {
                    "match_id": pred.match_id,
                    "home_team": pred.home_team,
                    "away_team": pred.away_team,
                    "league": pred.league,
                    "date": pred.date,
                    "home_win_prob": pred.home_win_prob,
                    "draw_prob": pred.draw_prob,
                    "away_win_prob": pred.away_win_prob,
                    "expected_home_goals": pred.expected_home_goals,
                    "expected_away_goals": pred.expected_away_goals,
                    "model_type": pred.model_type,
                    "confidence": pred.confidence,
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Predictions saved to {filename}")

    def save_betting_opportunities_to_csv(
        self, opportunities: List[BettingOpportunityDTO], filename: str
    ) -> None:
        if not opportunities:
            return

        data = []
        for opp in opportunities:
            data.append(
                {
                    "match_id": opp.match_id,
                    "home_team": opp.home_team,
                    "away_team": opp.away_team,
                    "league": opp.league,
                    "date": opp.date,
                    "bet_type": opp.bet_type,
                    "odds": opp.odds,
                    "stake": opp.stake,
                    "model_probability": opp.model_probability,
                    "market_probability": opp.market_probability,
                    "edge": opp.edge,
                    "expected_value": opp.expected_value,
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Betting opportunities saved to {filename}")
