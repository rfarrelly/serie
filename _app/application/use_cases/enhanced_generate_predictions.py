from datetime import date, datetime
from typing import List, Tuple

from domain.repositories.match_repository import MatchRepository
from domain.repositories.team_repository import TeamRepository
from domain.services.enhanced_betting_service import EnhancedBettingService
from domain.services.enhanced_prediction_service import EnhancedPredictionService
from domain.services.model_service import ModelService
from domain.services.ppi_service import PPIService
from domain.value_objects.timeframe import Timeframe

from ..dto.enhanced_prediction_dto import (
    EnhancedBettingOpportunityDTO,
    EnhancedPredictionResponse,
)


class EnhancedGeneratePredictionsUseCase:
    def __init__(
        self,
        match_repository: MatchRepository,
        team_repository: TeamRepository,
        prediction_service: EnhancedPredictionService,
        betting_service: EnhancedBettingService,
        model_service: ModelService,
        ppi_service: PPIService,
    ):
        self.match_repository = match_repository
        self.team_repository = team_repository
        self.prediction_service = prediction_service
        self.betting_service = betting_service
        self.model_service = model_service
        self.ppi_service = ppi_service

    def execute(
        self,
        league: str = None,
        season: str = None,
        start_date: date = None,
        end_date: date = None,
    ) -> Tuple[List[EnhancedPredictionResponse], List[EnhancedBettingOpportunityDTO]]:
        # Get upcoming matches
        upcoming_matches = self.match_repository.find_upcoming_matches()

        # Filter by criteria
        if league:
            upcoming_matches = [m for m in upcoming_matches if m.league == league]

        if start_date and end_date:
            timeframe = Timeframe(start_date, end_date)
            upcoming_matches = [
                m for m in upcoming_matches if timeframe.contains(m.date.date())
            ]

        if not upcoming_matches:
            print(f"No upcoming matches found for league: {league}")
            return [], []

        print(f"Found {len(upcoming_matches)} upcoming matches")

        # Get historical matches for model fitting
        historical_matches = self.match_repository.find_played_matches()

        if not historical_matches:
            print("No historical matches found for model fitting")
            return [], []

        print(f"Using {len(historical_matches)} historical matches for model fitting")

        # Fit model with historical data
        team_ratings = self.model_service.fit_ratings(historical_matches)
        print(f"Fitted ratings for {len(team_ratings)} teams")

        # Fit prediction models
        self.prediction_service.fit_adjustment_matrix(historical_matches)
        self.prediction_service.fit_mov_model(historical_matches, team_ratings)

        # Generate predictions
        predictions = []
        betting_opportunities = []

        for match in upcoming_matches:
            try:
                # Get team ratings
                home_ratings = self.model_service.get_team_rating(match.home_team)
                away_ratings = self.model_service.get_team_rating(match.away_team)

                # Generate predictions using all methods
                prediction_dict = self.prediction_service.predict_match_all_methods(
                    match,
                    home_ratings,
                    away_ratings,
                    historical_matches,
                    self.model_service.home_advantage,
                    self.model_service.away_adjustment,
                )

                # Calculate PPI data
                home_ppi_data = self.ppi_service.calculate_team_ppi(
                    match.home_team, historical_matches
                )
                away_ppi_data = self.ppi_service.calculate_team_ppi(
                    match.away_team, historical_matches
                )
                ppi_diff = abs(home_ppi_data.ppi - away_ppi_data.ppi)

                # Create enhanced prediction response
                pred_response = self._create_prediction_response(
                    match, prediction_dict, home_ppi_data, away_ppi_data, ppi_diff
                )

                # Betting analysis if odds available
                betting_analysis = None
                if match.odds:
                    betting_analysis = self.betting_service.analyze_predictions(
                        prediction_dict, match.odds
                    )

                    # Update prediction with betting info
                    pred_response = self._add_betting_info_to_prediction(
                        pred_response, betting_analysis, match.odds
                    )

                    # Check for betting opportunities
                    opportunities = (
                        self.betting_service.find_betting_opportunities_enhanced(
                            betting_analysis
                        )
                    )

                    for opp in opportunities:
                        opp_dto = self._create_betting_opportunity_dto(
                            opp, match, prediction_dict, betting_analysis, ppi_diff
                        )
                        betting_opportunities.append(opp_dto)
                        pred_response.is_betting_candidate = True

                predictions.append(pred_response)

            except Exception as e:
                print(
                    f"Error generating prediction for {match.home_team} vs {match.away_team}: {e}"
                )
                import traceback

                traceback.print_exc()
                continue

        print(f"Generated {len(predictions)} predictions")
        print(f"Found {len(betting_opportunities)} betting opportunities")

        return predictions, betting_opportunities

    def _create_prediction_response(
        self, match, prediction_dict, home_ppi_data, away_ppi_data, ppi_diff
    ) -> EnhancedPredictionResponse:
        """Create enhanced prediction response with all data"""

        # Get predictions by method
        poisson_pred = prediction_dict.get("poisson")
        zip_pred = prediction_dict.get("zip")
        mov_pred = prediction_dict.get("mov")

        return EnhancedPredictionResponse(
            match_id=zip_pred.match_id,
            home_team=match.home_team,
            away_team=match.away_team,
            league=match.league,
            date=match.date.isoformat(),
            week=match.week,
            # All method probabilities
            poisson_prob_h=poisson_pred.home_win_prob.value if poisson_pred else 0.0,
            poisson_prob_d=poisson_pred.draw_prob.value if poisson_pred else 0.0,
            poisson_prob_a=poisson_pred.away_win_prob.value if poisson_pred else 0.0,
            zip_prob_h=zip_pred.home_win_prob.value if zip_pred else 0.0,
            zip_prob_d=zip_pred.draw_prob.value if zip_pred else 0.0,
            zip_prob_a=zip_pred.away_win_prob.value if zip_pred else 0.0,
            mov_prob_h=mov_pred.home_win_prob.value if mov_pred else 0.0,
            mov_prob_d=mov_pred.draw_prob.value if mov_pred else 0.0,
            mov_prob_a=mov_pred.away_win_prob.value if mov_pred else 0.0,
            # Expected goals (from ZIP model as primary)
            lambda_home=zip_pred.expected_home_goals if zip_pred else 0.0,
            lambda_away=zip_pred.expected_away_goals if zip_pred else 0.0,
            # MOV prediction
            mov_prediction=(
                zip_pred.expected_home_goals - zip_pred.expected_away_goals
                if zip_pred
                else 0.0
            ),
            mov_std_error=1.0,  # Default
            # Odds (if available)
            sharp_odds_h=match.odds.home.value if match.odds else None,
            sharp_odds_d=match.odds.draw.value if match.odds else None,
            sharp_odds_a=match.odds.away.value if match.odds else None,
            soft_odds_h=(
                match.odds.home.value if match.odds else None
            ),  # Assuming same for now
            soft_odds_d=match.odds.draw.value if match.odds else None,
        )
