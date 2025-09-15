from typing import Dict, List, Optional

import pandas as pd
from config import END_DATE, TODAY, AppConfig, Leagues
from domains.data.services import PPICalculationService
from infrastructure.adapters.file_adapter import CSVFileAdapter
from utils.datetime_helpers import filter_date_range


class LeagueProcessingService:
    """Application service for processing league data and PPI calculations."""

    def __init__(
        self,
        ppi_service: PPICalculationService,
        file_adapter: CSVFileAdapter,
        config: AppConfig,
    ):
        self.ppi_service = ppi_service
        self.file_adapter = file_adapter
        self.config = config

    def process_latest_ppi_for_all_leagues(self) -> List[Dict]:
        """Process latest PPI for all leagues."""
        all_ppi_data = []
        failed_leagues = []

        for league in Leagues:
            try:
                league_processor = SingleLeagueProcessor(
                    league, self.config, self.ppi_service, self.file_adapter
                )

                ppi_data = league_processor.get_points_performance_index()
                if ppi_data:
                    all_ppi_data.extend(ppi_data)
                    print(f"Generated {len(ppi_data)} PPI records for {league.name}")
                else:
                    print(f"No PPI data for {league.name}")

            except Exception as e:
                print(f"Error processing {league.name}: {e}")
                failed_leagues.append(league.name)
                continue

        if failed_leagues:
            print(f"Failed leagues: {', '.join(failed_leagues)}")

        return all_ppi_data

    def process_historical_ppi_for_all_leagues(self) -> pd.DataFrame:
        """Process historical PPI data for all leagues."""
        files = list(self.config.fbref_data_dir.rglob("*.csv"))
        files = [f for f in files if f.is_file() and "unplayed" not in str(f)]

        all_ppi_data = []

        for file in files:
            try:
                fixtures = pd.read_csv(file, dtype={"Wk": int}).sort_values("Date")
                league_ppi = self.ppi_service.calculate_historical_ppi(fixtures)
                all_ppi_data.append(league_ppi)

            except Exception as e:
                print(f"Error processing historical PPI for {file}: {e}")
                continue

        if not all_ppi_data:
            raise ValueError("No historical PPI data could be processed")

        combined_ppi = pd.concat(all_ppi_data)

        # Handle terminated matches (like Reus)
        combined_ppi = combined_ppi[
            ~combined_ppi["Home"].eq("Reus") & ~combined_ppi["Away"].eq("Reus")
        ]

        print(f"Historical processor processed: {combined_ppi.shape[0]} records")
        return combined_ppi.sort_values("Date")


class SingleLeagueProcessor:
    """Processes data for a single league."""

    def __init__(
        self,
        league: Leagues,
        config: AppConfig,
        ppi_service: PPICalculationService,
        file_adapter: CSVFileAdapter,
    ):
        self.league = league
        self.config = config
        self.league_name = league.fbref_name
        self.ppi_service = ppi_service
        self.file_adapter = file_adapter
        self.fbref_dir = config.get_fbref_league_dir(self.league_name)

    def get_points_performance_index(self) -> Optional[List[Dict]]:
        """Calculate PPI for upcoming fixtures in this league."""
        try:
            # Load played and unplayed matches
            played_matches_df = self._load_played_matches()
            unplayed_matches_df = self._load_unplayed_matches()

            if played_matches_df.empty:
                print(f"No played matches found for {self.league_name}")
                return None

            # Filter fixtures by date range
            fixtures = filter_date_range(unplayed_matches_df, TODAY, END_DATE)

            if fixtures.empty:
                print(f"No fixtures in date range for {self.league_name}")
                return None

            # Calculate PPI
            return self.ppi_service.calculate_latest_ppi_for_league(
                self.league_name, played_matches_df, fixtures
            )

        except Exception as e:
            print(f"Error calculating PPI for {self.league_name}: {e}")
            return None

    def _load_played_matches(self) -> pd.DataFrame:
        """Load played matches for the league."""
        file_path = (
            self.fbref_dir / f"{self.league_name}_{self.config.current_season}.csv"
        )

        if not file_path.exists():
            return pd.DataFrame()

        try:
            return pd.read_csv(file_path, dtype={"Wk": int})
        except Exception as e:
            print(f"Error loading played matches for {self.league_name}: {e}")
            return pd.DataFrame()

    def _load_unplayed_matches(self) -> pd.DataFrame:
        """Load unplayed matches for the league."""
        file_path = (
            self.fbref_dir
            / f"unplayed_{self.league_name}_{self.config.current_season}.csv"
        )

        if not file_path.exists():
            return pd.DataFrame()

        try:
            return pd.read_csv(file_path, dtype={"Wk": int})
        except Exception as e:
            print(f"Error loading unplayed matches for {self.league_name}: {e}")
            return pd.DataFrame()
