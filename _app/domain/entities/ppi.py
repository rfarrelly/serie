from dataclasses import dataclass
from typing import Optional


@dataclass
class PPIData:
    """Points Performance Index data for a team"""

    team_name: str
    league: str
    ppg: float  # Points per game
    opp_ppg: float  # Opposition points per game
    ppi: float  # Points Performance Index (ppg * opp_ppg)

    @property
    def performance_indicator(self) -> str:
        if self.ppi > 6.0:
            return "Excellent"
        elif self.ppi > 4.0:
            return "Good"
        elif self.ppi > 2.0:
            return "Average"
        else:
            return "Poor"
