from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional

from ..entities.betting import BettingOpportunity, BettingResult
from ..entities.prediction import Prediction


class PredictionRepository(ABC):
    @abstractmethod
    def find_by_match_id(self, match_id: str) -> Optional[Prediction]:
        pass

    @abstractmethod
    def find_by_date_range(self, start_date: date, end_date: date) -> List[Prediction]:
        pass

    @abstractmethod
    def save(self, prediction: Prediction) -> None:
        pass

    @abstractmethod
    def save_all(self, predictions: List[Prediction]) -> None:
        pass


class BettingRepository(ABC):
    @abstractmethod
    def find_opportunities_by_date(self, target_date: date) -> List[BettingOpportunity]:
        pass

    @abstractmethod
    def find_results_by_date_range(
        self, start_date: date, end_date: date
    ) -> List[BettingResult]:
        pass

    @abstractmethod
    def save_opportunity(self, opportunity: BettingOpportunity) -> None:
        pass

    @abstractmethod
    def save_result(self, result: BettingResult) -> None:
        pass
