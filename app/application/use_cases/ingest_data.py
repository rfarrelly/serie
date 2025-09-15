from typing import List

import pandas as pd
from application.services.league_processing_service import LeagueProcessingService
from domains.data.entities import Match
from domains.data.repositories import MatchRepository
from domains.data.services import PPICalculationService
from infrastructure.adapters.file_adapter import CSVFileAdapter


class IngestDataUseCase:
    def __init__(
        self,
        match_repository: MatchRepository,
        ppi_service: PPICalculationService,
        file_adapter: CSVFileAdapter,
        league_processing_service: LeagueProcessingService = None,
    ):
        self.match_repository = match_repository
        self.ppi_service = ppi_service
        self.file_adapter = file_adapter
        self.league_processing_service = league_processing_service

    def execute_latest_ppi(self, leagues: List[str]) -> None:
        """Generate latest PPI data for all leagues using the processing service"""
        if self.league_processing_service:
            # Use the new service
            all_ppi_data = (
                self.league_processing_service.process_latest_ppi_for_all_leagues()
            )
        else:
            # Fallback to direct service usage
            all_ppi_data = []
            for league in leagues:
                try:
                    # Get fixtures for the league
                    fixtures = self._get_league_fixtures(league)

                    if fixtures:
                        ppi_records = self.ppi_service.calculate_ppi_for_fixtures(
                            league, fixtures
                        )
                        all_ppi_data.extend(ppi_records)
                        print(f"Generated {len(ppi_records)} PPI records for {league}")
                except Exception as e:
                    print(f"Error processing {league}: {e}")
                    continue

        if all_ppi_data:
            # Sort by PPI_Diff and save
            ppi_df = pd.DataFrame(all_ppi_data).sort_values("PPI_Diff")
            self.file_adapter.write_csv(ppi_df, "latest_ppi.csv")
            print(f"Saved {len(all_ppi_data)} total PPI records")

    def _get_league_fixtures(self, league: str) -> List[Match]:
        """Get upcoming fixtures for a league"""
        from config import END_DATE, TODAY

        return self.match_repository.get_by_date_range(TODAY, END_DATE)
