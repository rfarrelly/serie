from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..shared.exceptions import InsufficientDataException
from .entities import Match
from .repositories import MatchRepository


class PPICalculationService:
    """Domain service for Points Performance Index calculations."""

    def __init__(self, match_repository: MatchRepository):
        self.match_repository = match_repository

    def calculate_ppi_for_fixtures(
        self, league: str, fixtures: List[Match]
    ) -> List[Dict]:
        """Calculate PPI for upcoming fixtures based on historical data."""
        historical_matches = self.match_repository.get_by_league(league)
        played_matches = [m for m in historical_matches if m.is_played]

        if len(played_matches) < 10:
            raise InsufficientDataException(f"Not enough historical data for {league}")

        # Convert to DataFrame for PPI calculation
        matches_df = self._matches_to_dataframe(played_matches)
        fixtures_df = self._matches_to_dataframe(fixtures)

        return self._calculate_ppi_records(matches_df, fixtures_df, league)

    def calculate_historical_ppi(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate historical PPI data for all matches in a dataset."""
        teams = set(matches_df["Home"]).union(matches_df["Away"])

        # Calculate PPG data
        home_ppg, away_ppg, total_ppg = self._compute_ppg(matches_df)

        # Calculate PPI for each team
        ppi_df_list = []
        for team in teams:
            try:
                team_ppi_df = self._compute_points_performance_index(
                    team, matches_df, home_ppg, away_ppg, total_ppg
                ).sort_values("Date")

                # Shift PPI values to avoid look-ahead bias
                team_ppi_df[["OppPPG", "PPG", "PPI"]] = team_ppi_df[
                    ["OppPPG", "PPG", "PPI"]
                ].shift(periods=1, fill_value=0)
                ppi_df_list.append(team_ppi_df)
            except Exception as e:
                print(f"Error computing PPI for team {team}: {e}")
                continue

        if not ppi_df_list:
            raise InsufficientDataException("No PPI data could be calculated")

        # Combine all team PPI data
        ppi_df = pd.concat(ppi_df_list)

        # Pivot to get home/away PPI in same row
        pivot_cols = ["OppPPG", "PPG", "PPI"]
        ppi_df_wide = ppi_df.pivot_table(
            index=[
                "Wk",
                "League",
                "Season",
                "Day",
                "Date",
                "Home",
                "Away",
                "FTHG",
                "FTAG",
            ],
            columns="TeamType",
            values=pivot_cols,
        )

        ppi_df_wide.columns = [f"{side}{col}" for col, side in ppi_df_wide.columns]
        ppi_final = ppi_df_wide.reset_index()

        # Calculate PPI difference
        ppi_final["PPI_Diff"] = round(abs(ppi_final["hPPI"] - ppi_final["aPPI"]), 2)

        return ppi_final.sort_values("Date")

    def calculate_latest_ppi_for_league(
        self, league: str, played_matches_df: pd.DataFrame, fixtures_df: pd.DataFrame
    ) -> List[Dict]:
        """Calculate PPI for upcoming fixtures in a specific league."""
        if fixtures_df.empty:
            return []

        home_ppg, away_ppg, total_ppg = self._compute_ppg(played_matches_df)
        candidates = []

        for _, fixture in fixtures_df.iterrows():
            try:
                home_ppi_df = self._compute_points_performance_index(
                    fixture["Home"], played_matches_df, home_ppg, away_ppg, total_ppg
                )
                away_ppi_df = self._compute_points_performance_index(
                    fixture["Away"], played_matches_df, home_ppg, away_ppg, total_ppg
                )

                home_ppi = home_ppi_df.tail(1)["PPI"].values[0]
                away_ppi = away_ppi_df.tail(1)["PPI"].values[0]

                candidates.append(
                    {
                        "Wk": fixture["Wk"],
                        "Date": fixture["Date"],
                        "League": league,
                        "Home": fixture["Home"],
                        "Away": fixture["Away"],
                        "aOppPPG": away_ppi_df.tail(1)["OppPPG"].values[0],
                        "hOppPPG": home_ppi_df.tail(1)["OppPPG"].values[0],
                        "aPPG": away_ppi_df.tail(1)["PPG"].values[0],
                        "hPPG": home_ppi_df.tail(1)["PPG"].values[0],
                        "hPPI": home_ppi,
                        "aPPI": away_ppi,
                        "PPI_Diff": round(abs(home_ppi - away_ppi), 2),
                    }
                )
            except Exception as e:
                print(
                    f"Error computing PPI for {fixture['Home']} vs {fixture['Away']}: {e}"
                )
                continue

        return candidates

    def _matches_to_dataframe(self, matches: List[Match]) -> pd.DataFrame:
        """Convert match entities to DataFrame format."""
        data = []
        for match in matches:
            data.append(
                {
                    "Date": match.date,
                    "Home": match.home_team.name,
                    "Away": match.away_team.name,
                    "FTHG": match.home_goals,
                    "FTAG": match.away_goals,
                    "Wk": match.week,
                    "League": match.home_team.league,
                }
            )
        return pd.DataFrame(data)

    def _calculate_ppi_records(
        self, matches_df: pd.DataFrame, fixtures_df: pd.DataFrame, league: str
    ) -> List[Dict]:
        """Calculate PPI records for fixtures."""
        home_ppg, away_ppg, total_ppg = self._compute_ppg(matches_df)
        candidates = []

        for _, fixture in fixtures_df.iterrows():
            try:
                home_ppi_df = self._compute_points_performance_index(
                    fixture["Home"], matches_df, home_ppg, away_ppg, total_ppg
                )
                away_ppi_df = self._compute_points_performance_index(
                    fixture["Away"], matches_df, home_ppg, away_ppg, total_ppg
                )

                home_ppi = home_ppi_df.tail(1)["PPI"].values[0]
                away_ppi = away_ppi_df.tail(1)["PPI"].values[0]

                candidates.append(
                    {
                        "Wk": fixture["Wk"],
                        "Date": fixture["Date"],
                        "League": league,
                        "Home": fixture["Home"],
                        "Away": fixture["Away"],
                        "hPPI": home_ppi,
                        "aPPI": away_ppi,
                        "PPI_Diff": abs(home_ppi - away_ppi),
                    }
                )
            except Exception:
                continue

        return candidates

    def _compute_ppg(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute points per game statistics."""
        df = df.copy()

        # Calculate points for home and away teams
        df["HP"] = df.apply(
            lambda x: (
                3 if x["FTHG"] > x["FTAG"] else 1 if x["FTHG"] == x["FTAG"] else 0
            ),
            axis="columns",
        )
        df["AP"] = df.apply(
            lambda x: (
                3 if x["FTHG"] < x["FTAG"] else 1 if x["FTHG"] == x["FTAG"] else 0
            ),
            axis="columns",
        )

        # Pivot by week to get cumulative averages
        home_points = df.pivot(index="Home", columns="Wk", values="HP")
        away_points = df.pivot(index="Away", columns="Wk", values="AP")

        # Calculate expanding means (cumulative averages)
        home_ppg = home_points.T.expanding().mean().T.round(3).reset_index()
        away_ppg = away_points.T.expanding().mean().T.round(3).reset_index()

        # Total PPG combines home and away
        total_ppg = (
            home_points.combine_first(away_points)
            .T.expanding()
            .mean()
            .T.round(3)
            .reset_index()
            .rename({"Home": "Team"}, axis="columns")
        )

        return home_ppg, away_ppg, total_ppg

    def _compute_points_performance_index(
        self,
        team: str,
        df: pd.DataFrame,
        home_ppg: pd.DataFrame,
        away_ppg: pd.DataFrame,
        total_ppg: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute PPI for a specific team."""
        # Get home and away games for the team
        home_games = df[df["Home"] == team]
        away_games = df[df["Away"] == team]

        # Merge with opponent PPG data
        home_games = home_games.merge(away_ppg, left_on="Away", right_on="Away")
        away_games = away_games.merge(home_ppg, left_on="Home", right_on="Home")

        # Combine and sort by date
        combined = (
            pd.concat([home_games, away_games])
            .reset_index(drop=True)
            .sort_values("Date")
        )

        # Get week columns for expanding calculation
        weeks_columns = [x for x in combined.columns if isinstance(x, int)]

        # Calculate expanding mean of opponent PPG
        combined[weeks_columns] = combined[weeks_columns].expanding().mean().round(3)
        combined["OppPPG"] = np.diag(combined[weeks_columns])

        # Get team's own PPG
        team_ppg_data = total_ppg[total_ppg["Team"] == team][weeks_columns]
        if not team_ppg_data.empty:
            combined["PPG"] = team_ppg_data.values[0][: combined.shape[0]]
        else:
            combined["PPG"] = 0

        # Calculate PPI = OppPPG × TeamPPG
        combined["PPI"] = round(combined["OppPPG"] * combined["PPG"], 3)

        # Add team type indicator
        combined["TeamType"] = combined.apply(
            lambda x: "h" if team == x["Home"] else "a", axis=1
        )

        return combined.drop(weeks_columns, axis=1)
