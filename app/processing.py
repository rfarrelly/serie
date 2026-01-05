import pandas as pd
import rs
from config import END_DATE, TODAY, AppConfig, Leagues
from ingestion import DataIngestion
from utils.datetime_helpers import filter_date_range


class LeagueProcessor:
    def __init__(self, league: Leagues, config: AppConfig):
        self.league = league
        self.config = config
        self.league_name = league.fbref_name
        self.fbref_dir = config.get_fbref_league_dir(
            self.league.fbduk_id + "_" + self.league_name
            if self.league.is_extra
            else self.league_name
        )
        self.ingestion = DataIngestion(config)

    @property
    def played_matches_df(self):
        filename = f"{self.league_name}_{self.config.current_season}.csv"
        if self.league.is_extra:
            filename = f"{self.league.fbduk_id}_{self.league_name}_{self.config.current_season}.csv"
        return pd.read_csv(
            self.fbref_dir / filename,
            dtype={"Wk": int},
        )

    @property
    def unplayed_matches_df(self):
        filename = f"{self.league_name}_{self.config.current_season}.csv"
        if self.league.is_extra:
            filename = f"{self.league.fbduk_id}_{self.league_name}_{self.config.current_season}.csv"
        return pd.read_csv(
            self.fbref_dir / f"unplayed_{filename}",
            dtype={"Wk": int},
        )

    async def get_fbref_data(self, browser):
        await self.ingestion.get_fbref_data(
            league=self.league, season=self.config.current_season, browser=browser
        )

    def get_fbduk_data(self):
        self.ingestion.get_fbduk_data(
            league=self.league, season=self.config.current_season
        )

    # def get_points_performance_index(self) -> dict:
    #     fixtures = filter_date_range(self.unplayed_matches_df, TODAY, END_DATE)

    #     if fixtures.empty:
    #         print("No Fixtures for this date range")
    #         return None

    #     candidates = []

    #     for fixture in fixtures.itertuples(index=False):
    #          date, home_team, away_team = (
    #             fixture.Date,
    #             fixture.Home,
    #             fixture.Away
    #         )

    #         # try:
    #         #     ...
    #         # except:
    #         #     print(f"Error computing team metrics for {self.league_name} - {date}")
    #         #     print(f"Continuing ...")
    #         #     continue

    #         candidates.append(
    #             {
    #                 "Wk": week,
    #                 "Date": date,
    #                 "League": self.league_name,
    #                 "Home": home_team,
    #                 "Away": away_team,
    #                 "hOppPPG": home_opps_ppg,
    #                 "aOppPPG": away_opps_ppg,
    #                 "hPPG": home_ppg,
    #                 "aPPG": away_ppg,
    #                 "hPPI": latest_home_ppi,
    #                 "aPPI": latest_away_ppi,
    #                 "PPI_Diff": ppi_diff,
    #                 "hPPINorm": latest_ppi_home_norm,
    #                 "aPPINorm": latest_ppi_away_norm,
    #                 "PPINorm_Diff": ppi_norm_diff,
    #             }
    #         )

    #     candidates_df = pd.DataFrame(candidates)
    #     return candidates_df.to_dict(orient="records")


def get_historical_ppi(config: AppConfig) -> pd.DataFrame:
    print("Processing historical PPI")

    exclude_leagues = [league.fbref_name for league in Leagues if league.is_extra]

    files = [
        str(file)
        for file in config.fbref_data_dir.rglob("*.csv")
        if file.is_file()
        if "unplayed" not in str(file)
        if not any(exclude in str(file) for exclude in exclude_leagues)
    ]

    historical_metrics = []

    for file_path in files:
        print(f"Processing {file_path}")
        try:
            historical_metrics.append(rs.compute_historical_ppi(file_path))
        except Exception as e:
            print(f"Failed to process {file_path}, continuing ... {e}")
            continue

    historical_metrics = pd.concat(historical_metrics)

    print(f"Historical processor processed: {historical_metrics.shape[0]} records")
    return historical_metrics.sort_values("Date").reset_index(drop=True)
