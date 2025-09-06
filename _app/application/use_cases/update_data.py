from typing import Dict, List

from domain.repositories.match_repository import MatchRepository
from infrastructure.data_sources.fbduk_client import FBDukClient
from infrastructure.data_sources.fbref_client import FBRefClient


class UpdateDataUseCase:
    def __init__(
        self,
        fbref_client: FBRefClient,
        fbduk_client: FBDukClient,
        match_repository: MatchRepository,
    ):
        self.fbref_client = fbref_client
        self.fbduk_client = fbduk_client
        self.match_repository = match_repository

    def execute(self, leagues: List[Dict], season: str) -> Dict[str, bool]:
        results = {}

        for league_config in leagues:
            league_name = league_config["name"]
            fbref_id = league_config["fbref_id"]
            fbduk_id = league_config["fbduk_id"]

            try:
                # Fetch FBRef data
                fbref_data = self.fbref_client.fetch_league_data(
                    fbref_id, season, league_name
                )
                if fbref_data is not None:
                    # Process and save data
                    # This would convert to Match entities and save via repository
                    pass

                # Fetch FBDuk data
                fbduk_data = self.fbduk_client.fetch_main_league_data(fbduk_id, season)
                if fbduk_data is not None:
                    # Process and save odds data
                    pass

                results[league_name] = True

            except Exception as e:
                print(f"Error updating data for {league_name}: {e}")
                results[league_name] = False

        return results
