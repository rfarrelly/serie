from typing import List

from domain.repositories.match_repository import MatchRepository
from domain.repositories.team_repository import TeamRepository
from domain.services.betting_service import BettingService
from domain.services.model_service import ModelService
from domain.services.prediction_service import PredictionService
from domain.value_objects.timeframe import Timeframe

from ..dto.betting_dto import BettingOpportunityDTO
from ..dto.prediction_dto import PredictionRequest, PredictionResponse


class GeneratePredictionsUseCase:
    def __init__(
        self,
        match_repository: MatchRepository,
        team_repository: TeamRepository,
        prediction_service: PredictionService,
        betting_service: BettingService,
        model_service: ModelService,
    ):
        self.match_repository = match_repository
        self.team_repository = team_repository
        self.prediction_service = prediction_service
        self.betting_service = betting_service
        self.model_service = model_service

    def execute(
        self, request: PredictionRequest
    ) -> tuple[List[PredictionResponse], List[BettingOpportunityDTO]]:
        # Get upcoming matches
        upcoming_matches = self.match_repository.find_upcoming_matches()

        # Filter by request criteria
        if request.league:
            upcoming_matches = [
                m for m in upcoming_matches if m.league == request.league
            ]

        if request.start_date and request.end_date:
            timeframe = Timeframe(request.start_date, request.end_date)
            upcoming_matches = [
                m for m in upcoming_matches if timeframe.contains(m.date.date())
            ]

        # Get historical matches for model fitting
        historical_matches = self.match_repository.find_played_matches()

        # Fit model with historical data
        self.model_service.fit_ratings(historical_matches)

        # Generate predictions
        predictions = []
        betting_opportunities = []

        for match in upcoming_matches:
            try:
                # Get team ratings
                home_ratings = self.model_service.get_team_rating(match.home_team)
                away_ratings = self.model_service.get_team_rating(match.away_team)

                # Generate prediction
                prediction = self.prediction_service.predict_match(
                    match,
                    home_ratings,
                    away_ratings,
                    self.model_service.home_advantage,
                    request.prediction_method,
                )

                # Convert to DTO
                pred_response = PredictionResponse(
                    match_id=prediction.match_id,
                    home_team=prediction.home_team,
                    away_team=prediction.away_team,
                    league=match.league,
                    date=match.date.isoformat(),
                    home_win_prob=prediction.home_win_prob.value,
                    draw_prob=prediction.draw_prob.value,
                    away_win_prob=prediction.away_win_prob.value,
                    expected_home_goals=prediction.expected_home_goals,
                    expected_away_goals=prediction.expected_away_goals,
                    model_type=prediction.model_type,
                    confidence=prediction.confidence,
                )
                predictions.append(pred_response)

                # Find betting opportunities if odds available
                if match.odds:
                    opportunities = self.betting_service.find_betting_opportunities(
                        prediction, match.odds
                    )

                    for opp in opportunities:
                        opp_dto = BettingOpportunityDTO(
                            match_id=opp.match_id,
                            home_team=match.home_team,
                            away_team=match.away_team,
                            bet_type=opp.bet_type,
                            odds=opp.odds.value,
                            stake=opp.stake,
                            model_probability=opp.model_probability.value,
                            market_probability=opp.market_probability.value,
                            edge=opp.edge,
                            expected_value=opp.expected_value,
                            league=match.league,
                            date=match.date.isoformat(),
                        )
                        betting_opportunities.append(opp_dto)

            except Exception as e:
                print(
                    f"Error generating prediction for {match.home_team} vs {match.away_team}: {e}"
                )
                continue

        return predictions, betting_opportunities
