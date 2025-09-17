from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..shared.exceptions import InsufficientDataException
from .entities import Match
from .repositories import MatchRepository


class PPICalculationService:
    """Points Performance Index calculations following original methodology exactly."""

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
        """Calculate historical PPI data following original methodology exactly."""
        matches_df = matches_df.sort_values(["Date"]).reset_index(drop=True)
        teams = set(matches_df["Home"]).union(matches_df["Away"])

        # Calculate PPG data using original approach
        home_ppg, away_ppg, total_ppg = self._compute_ppg(matches_df)

        # Calculate PPI for each team
        ppi_df_list = []
        for team in teams:
            try:
                team_ppi_df = self._compute_points_performance_index(
                    team, matches_df, home_ppg, away_ppg, total_ppg
                ).sort_values("Date")

                # Shift PPI values to avoid look-ahead bias (original approach)
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

        # Pivot to get home/away PPI in same row (original approach)
        pivot_cols = ["OppPPG", "PPG", "PPI"]

        # Handle the Time column if it exists
        index_cols = [
            "Wk",
            "League",
            "Season",
            "Day",
            "Date",
            "Home",
            "Away",
            "FTHG",
            "FTAG",
        ]
        if "Time" in ppi_df.columns:
            index_cols.insert(5, "Time")  # Insert Time after Date

        # Only use columns that actually exist
        available_index_cols = [col for col in index_cols if col in ppi_df.columns]

        ppi_df_wide = ppi_df.pivot_table(
            index=available_index_cols,
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
        """Calculate PPI for upcoming fixtures following original methodology."""
        if fixtures_df.empty:
            return []

        # Calculate PPG data
        home_ppg, away_ppg, total_ppg = self._compute_ppg(played_matches_df)
        candidates = []

        for _, fixture in fixtures_df.iterrows():
            try:
                home_team = fixture["Home"]
                away_team = fixture["Away"]

                # Calculate PPI for both teams using original method
                home_ppi_df = self._compute_points_performance_index(
                    home_team, played_matches_df, home_ppg, away_ppg, total_ppg
                )
                away_ppi_df = self._compute_points_performance_index(
                    away_team, played_matches_df, home_ppg, away_ppg, total_ppg
                )

                # Get latest values (no shift for latest calculations)
                home_ppi_latest = home_ppi_df.tail(1)["PPI"].values[0]
                away_ppi_latest = away_ppi_df.tail(1)["PPI"].values[0]

                candidates.append(
                    {
                        "Wk": fixture["Wk"],
                        "Date": fixture["Date"],
                        "League": league,
                        "Home": home_team,
                        "Away": away_team,
                        "aOppPPG": away_ppi_df.tail(1)["OppPPG"].values[0],
                        "hOppPPG": home_ppi_df.tail(1)["OppPPG"].values[0],
                        "aPPG": away_ppi_df.tail(1)["PPG"].values[0],
                        "hPPG": home_ppi_df.tail(1)["PPG"].values[0],
                        "hPPI": home_ppi_latest,
                        "aPPI": away_ppi_latest,
                        "PPI_Diff": round(abs(home_ppi_latest - away_ppi_latest), 2),
                    }
                )

            except Exception as e:
                print(
                    f"Error computing PPI for {fixture['Home']} vs {fixture['Away']}: {e}"
                )
                continue

        return candidates

    def _compute_ppg(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute points per game statistics following original methodology exactly."""
        df = df.copy()

        # Calculate points for home and away teams (original logic)
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

        # Pivot by week to get cumulative averages (original approach)
        home_points = df.pivot(index="Home", columns="Wk", values="HP")
        away_points = df.pivot(index="Away", columns="Wk", values="AP")

        # Calculate expanding means (cumulative averages) - original approach
        home_ppg = home_points.T.expanding().mean().T.round(3).reset_index()
        away_ppg = away_points.T.expanding().mean().T.round(3).reset_index()

        # Total PPG combines home and away - original approach
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
        """Compute PPI for a specific team following original methodology exactly."""
        # Get home and away games for the team (original approach)
        home_games = df[df["Home"] == team].copy()
        away_games = df[df["Away"] == team].copy()

        # KEY INSIGHT FROM ORIGINAL CODE:
        # When team plays at home, we merge with away_ppg of the opponent
        # When team plays away, we merge with home_ppg of the opponent
        # This gives us the opponent's performance in the SAME CONTEXT as they will face our team

        home_games = home_games.merge(away_ppg, left_on="Away", right_on="Away")
        away_games = away_games.merge(home_ppg, left_on="Home", right_on="Home")

        # Combine and sort by date (original approach)
        combined = (
            pd.concat([home_games, away_games])
            .reset_index(drop=True)
            .sort_values("Date")
        )

        # Get week columns for expanding calculation (original approach)
        weeks_columns = [x for x in combined.columns if isinstance(x, int)]

        # Calculate expanding mean of opponent PPG (original methodology)
        # This creates expanding averages for each week column
        combined[weeks_columns] = combined[weeks_columns].expanding().mean().round(3)

        # Extract diagonal to get opponent PPG at each point in time (original approach)
        # This is the key insight - np.diag extracts the PPG for the actual week being played
        combined["OppPPG"] = np.diag(combined[weeks_columns])

        combined["PPG"] = total_ppg.loc[
            total_ppg["Team"] == team, weeks_columns
        ].values[0][-combined.shape[0] :]

        # NOTE:
        # OppPPG = average PPG of ALL opposition teams played at home OR away
        # TeamPPG = combined home AND away PPG for the target team
        combined["PPI"] = round(combined["OppPPG"] * combined["PPG"], 3)
        combined["TeamType"] = combined.apply(
            lambda x: "h" if team == x["Home"] else "a", axis=1
        )

        return combined.drop(weeks_columns, axis=1)

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
        # Calculate PPG data
        home_ppg, away_ppg, total_ppg = self._compute_ppg(matches_df)
        candidates = []

        for _, fixture in fixtures_df.iterrows():
            try:
                home_team = fixture["Home"]
                away_team = fixture["Away"]

                home_ppi_df = self._compute_points_performance_index(
                    home_team, matches_df, home_ppg, away_ppg, total_ppg
                )
                away_ppi_df = self._compute_points_performance_index(
                    away_team, matches_df, home_ppg, away_ppg, total_ppg
                )

                home_ppi = home_ppi_df.tail(1)["PPI"].values[0]
                away_ppi = away_ppi_df.tail(1)["PPI"].values[0]

                candidates.append(
                    {
                        "Wk": fixture["Wk"],
                        "Date": fixture["Date"],
                        "League": league,
                        "Home": home_team,
                        "Away": away_team,
                        "hPPI": home_ppi,
                        "aPPI": away_ppi,
                        "PPI_Diff": abs(home_ppi - away_ppi),
                    }
                )

            except Exception:
                continue

        return candidates
