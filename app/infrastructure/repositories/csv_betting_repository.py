from typing import List

from domains.betting.entities import BettingOpportunity, BettingResult
from domains.betting.repositories import BettingRepository

from ..adapters.file_adapter import CSVFileAdapter


class CSVBettingRepository(BettingRepository):
    def __init__(self, file_adapter: CSVFileAdapter):
        self.file_adapter = file_adapter

    def save_opportunities(self, opportunities: List[BettingOpportunity]) -> None:
        """Save betting opportunities with proper attribute access."""
        data = []
        for opp in opportunities:
            try:
                # Get the probability for the recommended outcome safely
                if opp.recommended_outcome == "H":
                    model_prob_value = float(opp.prediction.probabilities.home)
                elif opp.recommended_outcome == "D":
                    model_prob_value = float(opp.prediction.probabilities.draw)
                else:  # "A"
                    model_prob_value = float(opp.prediction.probabilities.away)

                # Flatten betting opportunity for CSV storage
                opportunity_data = {
                    "Date": (
                        opp.prediction.match.date.strftime("%Y-%m-%d")
                        if hasattr(opp.prediction.match.date, "strftime")
                        else str(opp.prediction.match.date)
                    ),
                    "League": opp.prediction.match.home_team.league,
                    "Home": opp.prediction.match.home_team.name,
                    "Away": opp.prediction.match.away_team.name,
                    "Bet_Type": opp.recommended_outcome,
                    "Edge": float(opp.edge),
                    "Fair_Odds_Selected": float(opp.fair_odds),
                    "Model_Prob": model_prob_value,
                    "B365H": float(opp.market_odds.home),
                    "B365D": float(opp.market_odds.draw),
                    "B365A": float(opp.market_odds.away),
                    "Is_Betting_Candidate": True,
                    # Add prediction data safely
                    "ZSD_Prob_H": float(opp.prediction.probabilities.home),
                    "ZSD_Prob_D": float(opp.prediction.probabilities.draw),
                    "ZSD_Prob_A": float(opp.prediction.probabilities.away),
                }
                data.append(opportunity_data)
            except Exception as e:
                print(f"Error saving betting opportunity: {e}")
                import traceback

                traceback.print_exc()
                continue

        if data:
            self.file_adapter.write_dict_list(data, "zsd_betting_candidates.csv")
            print(f"Saved {len(data)} betting opportunities")
        else:
            print("No betting opportunities to save")

    def save_results(self, results: List[BettingResult]) -> None:
        # Implementation for saving betting results
        pass

    def get_latest_opportunities(self) -> List[BettingOpportunity]:
        # Implementation would read from CSV and convert back
        return []
