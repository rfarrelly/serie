# application/workflows/prediction_workflow.py
"""
Complete Prediction Workflow - Phase 3 Application Layer

Replaces scattered prediction logic with clean, orchestrated workflow.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd
from domain.entities.match import Fixture
from domain.repositories import FixtureRepository, MatchRepository
from domain.services.edge_calculator import EdgeCalculator
from domain.services.ppi_calculator import PPICalculator
from shared.exceptions import InsufficientDataException
from shared.types.common_types import LeagueName, Season


@dataclass
class PredictionWorkflowConfig:
    """Configuration for prediction workflow"""

    min_betting_edge: Decimal = Decimal("0.02")
    max_ppi_differential: Decimal = Decimal("0.5")
    model_weight: Decimal = Decimal("0.1")
    historical_season: Season = Season("2023-2024")
    min_matches_for_ppi: int = 5
    max_predictions: int = 50


@dataclass
class PredictionWorkflowResult:
    """Complete result of prediction workflow"""

    enhanced_predictions: List["EnhancedPrediction"]
    betting_candidates: List["EnhancedPrediction"]
    summary_stats: Dict[str, Any]
    execution_time: float
    warnings: List[str]
    errors: List[str]


@dataclass
class EnhancedPrediction:
    """Enhanced prediction with all analysis"""

    fixture: Fixture
    home_ppi: Optional["PPICalculationResult"] = None
    away_ppi: Optional["PPICalculationResult"] = None
    ppi_differential: Optional[Decimal] = None
    edge_analysis: Optional["EdgeCalculationResult"] = None
    betting_opportunity: Optional["BettingOpportunity"] = None
    recommendation: str = "ANALYZE"
    confidence_score: Decimal = Decimal("0")


class GenerateCompletePredictionsWorkflow:
    """
    Complete prediction workflow orchestrating all domain services.

    This replaces your scattered prediction logic with a clean,
    testable, maintainable workflow.
    """

    def __init__(
        self,
        match_repository: MatchRepository,
        fixture_repository: FixtureRepository,
        ppi_calculator: PPICalculator,
        edge_calculator: EdgeCalculator,
        config: PredictionWorkflowConfig,
    ):
        self._match_repo = match_repository
        self._fixture_repo = fixture_repository
        self._ppi_calculator = ppi_calculator
        self._edge_calculator = edge_calculator
        self._config = config

    def execute(
        self, league_filter: Optional[LeagueName] = None
    ) -> PredictionWorkflowResult:
        """
        Execute complete prediction workflow.

        Process:
        1. Load upcoming fixtures with predictions
        2. Load historical data for PPI analysis
        3. Calculate PPI for all teams
        4. Analyze betting edges
        5. Generate recommendations
        6. Rank and filter results
        """
        start_time = datetime.now()
        warnings = []
        errors = []

        try:
            # Step 1: Load fixtures to analyze
            print("📊 Loading upcoming fixtures...")
            fixtures = self._load_fixtures(league_filter)

            if not fixtures:
                return PredictionWorkflowResult(
                    enhanced_predictions=[],
                    betting_candidates=[],
                    summary_stats={"total_fixtures": 0},
                    execution_time=0.0,
                    warnings=["No fixtures available for prediction"],
                    errors=[],
                )

            print(f"   Found {len(fixtures)} fixtures to analyze")

            # Step 2: Load historical data
            print("📚 Loading historical data for PPI analysis...")
            historical_matches = self._load_historical_data(fixtures)
            print(f"   Loaded {len(historical_matches)} historical matches")

            # Step 3: Generate enhanced predictions
            print("🔮 Generating enhanced predictions...")
            enhanced_predictions = []

            for i, fixture in enumerate(fixtures[: self._config.max_predictions]):
                if i % 10 == 0:
                    print(f"   Progress: {i}/{len(fixtures)} fixtures analyzed")

                try:
                    prediction = self._analyze_fixture(
                        fixture, historical_matches, warnings
                    )
                    enhanced_predictions.append(prediction)
                except Exception as e:
                    error_msg = f"Error analyzing {fixture.home_team} vs {fixture.away_team}: {str(e)}"
                    errors.append(error_msg)
                    print(f"   ⚠️ {error_msg}")

            print(f"✅ Generated {len(enhanced_predictions)} predictions")

            # Step 4: Rank and filter
            print("🎯 Ranking predictions and identifying betting opportunities...")
            ranked_predictions = self._rank_predictions(enhanced_predictions)
            betting_candidates = self._identify_betting_candidates(ranked_predictions)

            # Step 5: Generate summary
            summary_stats = self._generate_summary_stats(
                ranked_predictions, betting_candidates
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            return PredictionWorkflowResult(
                enhanced_predictions=ranked_predictions,
                betting_candidates=betting_candidates,
                summary_stats=summary_stats,
                execution_time=execution_time,
                warnings=warnings,
                errors=errors,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            errors.append(f"Workflow failed: {str(e)}")

            return PredictionWorkflowResult(
                enhanced_predictions=[],
                betting_candidates=[],
                summary_stats={"error": str(e)},
                execution_time=execution_time,
                warnings=warnings,
                errors=errors,
            )

    def _load_fixtures(self, league_filter: Optional[LeagueName]) -> List[Fixture]:
        """Load upcoming fixtures with predictions"""
        try:
            fixtures = self._fixture_repo.get_fixtures_with_predictions()

            if league_filter:
                fixtures = [f for f in fixtures if f.league == league_filter]

            # Sort by date
            fixtures.sort(key=lambda f: f.date)

            return fixtures

        except Exception as e:
            # Fall back to basic fixtures if enhanced not available
            try:
                fixtures = self._fixture_repo.get_upcoming_fixtures(league_filter)
                return fixtures
            except Exception as fallback_error:
                raise InsufficientDataException(
                    f"Could not load fixtures: {e}. Fallback also failed: {fallback_error}"
                )

    def _load_historical_data(self, fixtures: List[Fixture]) -> List["Match"]:
        """Load historical data for PPI analysis"""
        historical_matches = []
        leagues_to_load = set(f.league for f in fixtures)

        for league in leagues_to_load:
            try:
                league_matches = self._match_repo.get_matches_by_league_and_season(
                    league, self._config.historical_season
                )
                historical_matches.extend(league_matches)
            except Exception as e:
                print(f"   ⚠️ Could not load {league} historical data: {e}")

        if not historical_matches:
            raise InsufficientDataException(
                "No historical data available for PPI analysis"
            )

        return historical_matches

    def _analyze_fixture(
        self, fixture: Fixture, historical_matches: List["Match"], warnings: List[str]
    ) -> EnhancedPrediction:
        """Analyze single fixture with all domain services"""

        prediction = EnhancedPrediction(fixture=fixture)

        # Calculate PPI if sufficient data
        try:
            home_ppi = self._ppi_calculator.calculate_team_ppi(
                fixture.home_team, historical_matches
            )
            away_ppi = self._ppi_calculator.calculate_team_ppi(
                fixture.away_team, historical_matches
            )

            if (
                home_ppi.matches_analyzed >= self._config.min_matches_for_ppi
                and away_ppi.matches_analyzed >= self._config.min_matches_for_ppi
            ):
                prediction.home_ppi = home_ppi
                prediction.away_ppi = away_ppi
                prediction.ppi_differential = abs(
                    home_ppi.ppi_value - away_ppi.ppi_value
                )

        except Exception as e:
            warnings.append(
                f"PPI calculation failed for {fixture.home_team} vs {fixture.away_team}: {e}"
            )

        # Analyze betting opportunity if complete data available
        try:
            if self._has_complete_betting_data(fixture):
                edge_analysis, betting_opportunity = self._analyze_betting_opportunity(
                    fixture
                )
                prediction.edge_analysis = edge_analysis
                prediction.betting_opportunity = betting_opportunity

        except Exception as e:
            warnings.append(
                f"Edge analysis failed for {fixture.home_team} vs {fixture.away_team}: {e}"
            )

        # Generate recommendation and confidence
        prediction.recommendation = self._generate_recommendation(prediction)
        prediction.confidence_score = self._calculate_confidence_score(prediction)

        return prediction

    def _has_complete_betting_data(self, fixture: Fixture) -> bool:
        """Check if fixture has complete data for betting analysis"""
        from shared.types.common_types import BookmakerType

        # Need model predictions
        if not fixture.model_predictions:
            return False

        # Need both sharp and soft odds
        sharp_odds = fixture.market_odds.get(BookmakerType.PINNACLE)
        soft_odds = fixture.market_odds.get(BookmakerType.BET365)

        return sharp_odds is not None and soft_odds is not None

    def _analyze_betting_opportunity(self, fixture: Fixture):
        """Analyze betting opportunity using domain services"""
        from shared.types.common_types import BookmakerType

        # Get best model prediction
        model_prediction = None
        for model_type in ["ZIP", "ZSD", "Poisson", "MOV"]:
            if model_type in fixture.model_predictions:
                model_prediction = fixture.model_predictions[model_type]
                break

        if not model_prediction:
            raise ValueError("No model predictions available")

        # Get odds
        sharp_odds = fixture.market_odds[BookmakerType.PINNACLE]
        soft_odds = fixture.market_odds[BookmakerType.BET365]

        # Calculate edge
        edge_analysis = self._edge_calculator.calculate_comprehensive_edge(
            model_prediction, sharp_odds, soft_odds
        )

        # Create betting opportunity if significant
        betting_opportunity = None
        if edge_analysis.best_edge >= self._config.min_betting_edge:
            no_vig_result = self._edge_calculator.remove_bookmaker_margin(sharp_odds)
            weighted_probs = self._edge_calculator._combine_probabilities(
                model_prediction, no_vig_result.no_vig_probabilities
            )

            betting_opportunity = self._edge_calculator.create_betting_opportunity(
                edge_analysis, sharp_odds, soft_odds, weighted_probs
            )

        return edge_analysis, betting_opportunity

    def _generate_recommendation(self, prediction: EnhancedPrediction) -> str:
        """Generate recommendation based on analysis"""

        # Strong betting opportunity
        if (
            prediction.betting_opportunity
            and prediction.betting_opportunity.edge >= Decimal("0.05")
        ):
            return (
                f"STRONG BET: {prediction.betting_opportunity.bet_type.value} "
                f"({prediction.betting_opportunity.edge:.3f} edge)"
            )

        # Moderate betting opportunity
        if prediction.betting_opportunity:
            return (
                f"BET: {prediction.betting_opportunity.bet_type.value} "
                f"({prediction.betting_opportunity.edge:.3f} edge)"
            )

        # PPI-based recommendation
        if (
            prediction.ppi_differential
            and prediction.ppi_differential <= self._config.max_ppi_differential
        ):
            return f"CLOSE MATCH: PPI diff {prediction.ppi_differential:.3f} - analyze for value"

        # Model edge without betting threshold
        if prediction.edge_analysis and prediction.edge_analysis.best_edge > 0:
            return (
                f"WATCH: {prediction.edge_analysis.best_bet_type.value} has model edge "
                f"({prediction.edge_analysis.best_edge:.3f}) but below betting threshold"
            )

        return "ANALYZE: No clear opportunity identified"

    def _calculate_confidence_score(self, prediction: EnhancedPrediction) -> Decimal:
        """Calculate confidence score for ranking"""
        score = Decimal("0")

        # Betting opportunity contributes most
        if prediction.betting_opportunity:
            score += prediction.betting_opportunity.edge * Decimal("10")

        # PPI differential contributes (lower = better for close matches)
        if prediction.ppi_differential:
            if prediction.ppi_differential <= self._config.max_ppi_differential:
                score += (
                    self._config.max_ppi_differential - prediction.ppi_differential
                ) * Decimal("2")

        # Model edge contributes
        if prediction.edge_analysis:
            score += max(prediction.edge_analysis.best_edge, Decimal("0"))

        return score

    def _rank_predictions(
        self, predictions: List[EnhancedPrediction]
    ) -> List[EnhancedPrediction]:
        """Rank predictions by confidence score"""
        return sorted(predictions, key=lambda p: p.confidence_score, reverse=True)

    def _identify_betting_candidates(
        self, predictions: List[EnhancedPrediction]
    ) -> List[EnhancedPrediction]:
        """Identify predictions that are betting candidates"""
        return [
            p
            for p in predictions
            if p.betting_opportunity
            and p.betting_opportunity.is_significant(self._config.min_betting_edge)
        ]

    def _generate_summary_stats(
        self,
        predictions: List[EnhancedPrediction],
        betting_candidates: List[EnhancedPrediction],
    ) -> Dict[str, Any]:
        """Generate summary statistics"""

        stats = {
            "total_predictions": len(predictions),
            "betting_candidates": len(betting_candidates),
            "leagues_analyzed": len(set(p.fixture.league for p in predictions)),
        }

        if betting_candidates:
            edges = [p.betting_opportunity.edge for p in betting_candidates]
            stats.update(
                {
                    "avg_betting_edge": float(sum(edges) / len(edges)),
                    "max_betting_edge": float(max(edges)),
                    "bet_type_distribution": {
                        bt.value: len(
                            [
                                p
                                for p in betting_candidates
                                if p.betting_opportunity.bet_type == bt
                            ]
                        )
                        for bt in set(
                            p.betting_opportunity.bet_type for p in betting_candidates
                        )
                    },
                }
            )

        # PPI analysis
        ppi_predictions = [p for p in predictions if p.ppi_differential is not None]
        if ppi_predictions:
            ppi_diffs = [float(p.ppi_differential) for p in ppi_predictions]
            stats.update(
                {
                    "predictions_with_ppi": len(ppi_predictions),
                    "avg_ppi_differential": sum(ppi_diffs) / len(ppi_diffs),
                    "close_matches": len(
                        [
                            p
                            for p in ppi_predictions
                            if p.ppi_differential <= self._config.max_ppi_differential
                        ]
                    ),
                }
            )

        return stats
