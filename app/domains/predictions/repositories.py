from abc import ABC, abstractmethod
from typing import List, Optional

from .entities import ModelPerformance, Prediction


class PredictionRepository(ABC):
    @abstractmethod
    def save_predictions(self, predictions: List[Prediction]) -> None:
        pass

    @abstractmethod
    def get_latest_predictions(self) -> List[Prediction]:
        pass

    @abstractmethod
    def get_performance_metrics(self, model_type: str) -> Optional[ModelPerformance]:
        pass
