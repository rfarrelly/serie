# application/use_cases/get_betting_opportunities.py
from decimal import Decimal
from typing import List

from domain.entities.match import Fixture
from domain.repositories import FixtureRepository


class GetBettingOpportunitiesUseCase:
    """Use case to retrieve current betting opportunities"""

    def __init__(self, fixture_repository: FixtureRepository):
        self._fixture_repository = fixture_repository

    def execute(self, min_edge: float = 0.02) -> List[Fixture]:
        """Get all fixtures identified as betting candidates"""
        return self._fixture_repository.get_betting_candidates(min_edge)

    def get_significant_opportunities(self, min_edge: float = 0.05) -> List[Fixture]:
        """Get only high-confidence betting opportunities"""
        candidates = self._fixture_repository.get_betting_candidates(min_edge)

        # Additional domain filtering could go here
        return [
            fixture
            for fixture in candidates
            if fixture.betting_opportunities
            and any(
                opp.is_significant(Decimal(str(min_edge)))
                for opp in fixture.betting_opportunities
            )
        ]
