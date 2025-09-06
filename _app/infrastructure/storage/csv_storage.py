from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from domain.entities.match import Match
from domain.entities.prediction import Prediction
from domain.entities.team import Team, TeamRatings
from domain.repositories.match_repository import MatchRepository
from domain.repositories.prediction_repository import PredictionRepository
from domain.repositories.team_repository import TeamRatingsRepository, TeamRepository
from domain.value_objects.odds import Odds, OddsSet
from domain.value_objects.timeframe import Season, Timeframe


class CSVMatchRepository(MatchRepository):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)

    def find_by_teams(self, home_team: str, away_team: str) -> List[Match]:
        matches = []
        for csv_file in self.data_dir.rglob("*.csv"):
            if "unplayed" in csv_file.name:
                continue
            try:
                df = pd.read_csv(csv_file)
                team_matches = df[(df["Home"] == home_team) & (df["Away"] == away_team)]
                matches.extend(self._df_to_matches(team_matches))
            except Exception:
                continue
        return matches

    def find_by_league_and_season(self, league: str, season: Season) -> List[Match]:
        matches = []
        for csv_file in self.data_dir.rglob(f"*{league}*{season.name}*.csv"):
            try:
                df = pd.read_csv(csv_file)
                matches.extend(self._df_to_matches(df))
            except Exception:
                continue
        return matches

    def find_by_timeframe(self, timeframe: Timeframe) -> List[Match]:
        matches = []
        for csv_file in self.data_dir.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                df["Date"] = pd.to_datetime(df["Date"])
                filtered_df = df[
                    (df["Date"].dt.date >= timeframe.start_date)
                    & (df["Date"].dt.date <= timeframe.end_date)
                ]
                matches.extend(self._df_to_matches(filtered_df))
            except Exception:
                continue
        return matches

    def find_played_matches(self) -> List[Match]:
        matches = []
        for csv_file in self.data_dir.rglob("*.csv"):
            if "unplayed" in csv_file.name:
                continue
            try:
                df = pd.read_csv(csv_file)
                played_df = df.dropna(subset=["FTHG", "FTAG"])
                matches.extend(self._df_to_matches(played_df))
            except Exception:
                continue
        return matches

    def find_upcoming_matches(self) -> List[Match]:
        matches = []
        for csv_file in self.data_dir.rglob("*unplayed*.csv"):
            try:
                df = pd.read_csv(csv_file)
                matches.extend(self._df_to_matches(df))
            except Exception:
                continue
        return matches

    def save(self, match: Match) -> None:
        # Implementation would save single match
        pass

    def save_all(self, matches: List[Match]) -> None:
        # Implementation would save all matches
        pass

    def _df_to_matches(self, df: pd.DataFrame) -> List[Match]:
        matches = []

        for _, row in df.iterrows():
            try:
                # Parse date
                try:
                    match_date = pd.to_datetime(row["Date"])
                except (ValueError, TypeError):
                    print(f"Invalid date format: {row.get('Date', 'N/A')}")
                    continue

                # Parse odds if available
                odds = None
                if all(col in df.columns for col in ["PSH", "PSD", "PSA"]):
                    try:
                        if not (
                            pd.isna(row["PSH"])
                            or pd.isna(row["PSD"])
                            or pd.isna(row["PSA"])
                        ):
                            psh, psd, psa = (
                                float(row["PSH"]),
                                float(row["PSD"]),
                                float(row["PSA"]),
                            )
                            if all(odd > 1.0 for odd in [psh, psd, psa]):
                                odds = OddsSet(
                                    home=Odds(psh), draw=Odds(psd), away=Odds(psa)
                                )
                    except (ValueError, TypeError):
                        pass  # Skip invalid odds

                # Parse goals (may be NaN for upcoming matches)
                home_goals = None
                away_goals = None
                if "FTHG" in df.columns and "FTAG" in df.columns:
                    try:
                        if pd.notna(row["FTHG"]) and pd.notna(row["FTAG"]):
                            home_goals = int(float(row["FTHG"]))
                            away_goals = int(float(row["FTAG"]))
                    except (ValueError, TypeError):
                        pass  # Keep as None for upcoming matches

                # Parse week
                week = None
                if "Wk" in df.columns:
                    try:
                        if pd.notna(row["Wk"]):
                            week = int(float(row["Wk"]))
                    except (ValueError, TypeError):
                        pass

                match = Match(
                    home_team=str(row["Home"]).strip(),
                    away_team=str(row["Away"]).strip(),
                    date=match_date,
                    league=str(row.get("League", "Unknown")).strip(),
                    season=str(row.get("Season", "Unknown")).strip(),
                    week=week,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    odds=odds,
                )

                matches.append(match)

            except Exception as e:
                print(f"Error processing match row: {e}")
                print(f"Row data: {dict(row)}")
                continue

        return matches


class CSVTeamRepository(TeamRepository):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.mappings_file = Path("team_name_dictionary.csv")

    def find_by_name(self, name: str) -> Optional[Team]:
        # Implementation to find team by name
        return Team(name=name, league="Unknown", season="Unknown")

    def find_by_league(self, league: str) -> List[Team]:
        teams = set()
        for csv_file in self.data_dir.rglob(f"*{league}*.csv"):
            try:
                df = pd.read_csv(csv_file)
                teams.update(df["Home"].unique())
                teams.update(df["Away"].unique())
            except Exception:
                continue
        return [Team(name=name, league=league, season="Unknown") for name in teams]

    def find_all(self) -> List[Team]:
        teams = set()
        for csv_file in self.data_dir.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                teams.update(df["Home"].unique())
                teams.update(df["Away"].unique())
            except Exception:
                continue
        return [Team(name=name, league="Unknown", season="Unknown") for name in teams]

    def save(self, team: Team) -> None:
        pass

    def get_name_mappings(self) -> Dict[str, str]:
        if not self.mappings_file.exists():
            return {}
        try:
            df = pd.read_csv(self.mappings_file)
            return dict(zip(df["fbduk"], df["fbref"]))
        except Exception:
            return {}

    def save_name_mappings(self, mappings: Dict[str, str]) -> None:
        df = pd.DataFrame([{"fbduk": k, "fbref": v} for k, v in mappings.items()])
        df.to_csv(self.mappings_file, index=False)


class CSVTeamRatingsRepository(TeamRatingsRepository):
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_dir.mkdir(exist_ok=True)

    def find_by_team(self, team_name: str) -> Optional[TeamRatings]:
        # Implementation to find ratings for specific team
        return None

    def find_all(self) -> List[TeamRatings]:
        # Implementation to find all ratings
        return []

    def save(self, ratings: TeamRatings) -> None:
        pass

    def save_all(self, ratings: List[TeamRatings]) -> None:
        pass


class CSVPredictionRepository(PredictionRepository):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def find_by_match_id(self, match_id: str) -> Optional[Prediction]:
        return None

    def find_by_date_range(self, start_date: date, end_date: date) -> List[Prediction]:
        return []

    def save(self, prediction: Prediction) -> None:
        pass

    def save_all(self, predictions: List[Prediction]) -> None:
        df_data = []
        for pred in predictions:
            df_data.append(
                {
                    "match_id": pred.match_id,
                    "home_team": pred.home_team,
                    "away_team": pred.away_team,
                    "home_win_prob": pred.home_win_prob.value,
                    "draw_prob": pred.draw_prob.value,
                    "away_win_prob": pred.away_win_prob.value,
                    "expected_home_goals": pred.expected_home_goals,
                    "expected_away_goals": pred.expected_away_goals,
                    "model_type": pred.model_type,
                }
            )

        df = pd.DataFrame(df_data)
        output_file = (
            self.output_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        df.to_csv(output_file, index=False)
