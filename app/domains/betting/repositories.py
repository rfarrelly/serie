from abc import ABC, abstractmethod
from typing import List

from .entities import BettingOpportunity, BettingResult


class BettingRepository(ABC):
    @abstractmethod
    def save_opportunities(self, opportunities: List[BettingOpportunity]) -> None:
        pass

    @abstractmethod
    def save_results(self, results: List[BettingResult]) -> None:
        pass

    @abstractmethod
    def get_latest_opportunities(self) -> List[BettingOpportunity]:
        pass
