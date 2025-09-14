from typing import List

from domains.betting.entities import BettingOpportunity, BettingResult
from domains.betting.repositories import BettingRepository

from ..adapters.file_adapter import CSVFileAdapter


class CSVBettingRepository(BettingRepository):
    def __init__(self, file_adapter: CSVFileAdapter):
        self.file_adapter = file_adapter

    def save_opportunities(self, opportunities: List[BettingOpportunity]) -> None:
        data = []
        for opp in opportunities:
            # Flatten betting opportunity for CSV storage
            opportunity_data = {
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
                "B365H": float(opp.market_odds.home),
                "B365D": float(opp.market_odds.draw),
                "B365A": float(opp.market_odds.away),
                "Is_Betting_Candidate": True,
                # Add prediction data
                "ZSD_Prob_H": float(opp.prediction.probabilities.home),
                "ZSD_Prob_D": float(opp.prediction.probabilities.draw),
                "ZSD_Prob_A": float(opp.prediction.probabilities.away),
            }
            data.append(opportunity_data)

        self.file_adapter.write_dict_list(data, "zsd_betting_candidates.csv")

    def save_results(self, results: List[BettingResult]) -> None:
        # Implementation for saving betting results
        pass

    def get_latest_opportunities(self) -> List[BettingOpportunity]:
        # Implementation would read from CSV and convert back
        return []
