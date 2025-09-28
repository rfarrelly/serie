import pandas as pd
import pytest

from app.metrics import TeamMetrics

from .match_schedule_generator import generate_schedule

teams = ["TeamA", "TeamB", "TeamC", "TeamD", "TeamE", "TeamF"]


@pytest.fixture(scope="module")
def schedule_df():
    """Fixture that generates the match schedule once per test module."""
    return generate_schedule(teams, start_date="2025-08-09", seed=42)


def test_season_metrics_computes_metrics_tables(schedule_df: pd.DataFrame):
    team_metrics = TeamMetrics(team_name="TeamA", matches=schedule_df.copy())

    # home_played_opposition_teams = season_metrics.home_played_opposition_teams
    # away_played_opposition_teams = season_metrics.away_played_opposition_teams
    # team_total_weekly_ppg = season_metrics.team_total_weekly_ppg()
    # opposition_away_weekly_ppg = season_metrics.opposition_away_weekly_ppg()
    # opposition_home_weekly_ppg = season_metrics.opposition_home_weekly_ppg()
    # opposition_home_away_weekly_mean_ppg = (
    #     season_metrics.opposition_home_away_weekly_mean_ppg()
    # )
    # points_performance_index = team_metrics.points_performance_index()
    latest_points_performance_index = team_metrics.latest_points_performance_index
    breakpoint()
