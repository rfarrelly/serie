from typing import Dict, List

from domains.betting.repositories import BettingRepository
from domains.betting.services import BettingAnalysisService
from domains.predictions.entities import Prediction
from infrastructure.adapters.file_adapter import CSVFileAdapter


class CalculateBettingEdgesUseCase:
    def __init__(
        self,
        betting_service: BettingAnalysisService,
        betting_repository: BettingRepository,
        file_adapter: CSVFileAdapter,
    ):
        self.betting_service = betting_service
        self.betting_repository = betting_repository
        self.file_adapter = file_adapter

    def execute(self, predictions: List[Prediction]) -> List[Dict]:
        # Load odds data
        odds_data = self._load_odds_data()

        # Analyze predictions for betting opportunities
        opportunities = self.betting_service.analyze_predictions(predictions, odds_data)

        # Save opportunities
        self.betting_repository.save_opportunities(opportunities)

        # Convert to dictionary format
        opportunity_dicts = []
        for opp in opportunities:
            opp_dict = {
                "Date": opp.prediction.match.date,
                "League": opp.prediction.match.home_team.league,
                "Home": opp.prediction.match.home_team.name,
                "Away": opp.prediction.match.away_team.name,
                "Bet_Type": opp.recommended_outcome,
                "Edge": float(opp.edge),
                "Fair_Odds_Selected": float(opp.fair_odds),
                "Model_Prob": float(
                    getattr(
                        opp.prediction.probabilities, opp.recommended_outcome.lower()
                    )
                ),
                "Soft_Odds": float(
                    getattr(opp.market_odds, opp.recommended_outcome.lower())
                ),
                "Is_Betting_Candidate": True,
            }
            opportunity_dicts.append(opp_dict)

        return opportunity_dicts

    def _load_odds_data(self) -> Dict:
        """Load odds data from CSV files"""
        odds_dict = {}

        # Try to load from fixtures with odds
        if self.file_adapter.file_exists("fixtures_ppi_and_odds.csv"):
            df = self.file_adapter.read_csv("fixtures_ppi_and_odds.csv")
            for _, row in df.iterrows():
                key = f"{row['Home']}_{row['Away']}_{row['Date']}"
                odds_dict[key] = row.to_dict()

        return odds_dict
