import pandas as pd
import pandas.testing as pdt
import pytest

from app.metrics import TeamMetrics

from .match_schedule_generator import generate_schedule


@pytest.fixture(scope="module")
def schedule_df():
    """Fixture that generates the match schedule once per test module."""
    teams = ["TeamA", "TeamB", "TeamC", "TeamD", "TeamE", "TeamF"]
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
    # latest_points_performance_index = team_metrics.latest_points_performance_index
    metrics = team_metrics.team_home_metrics()


def test_ppi_correctness_for_sample_league():
    matches = pd.read_csv("tests/National-League_2025-2026.csv")

    soccerstats_reference_ppi = (
        pd.DataFrame(
            {
                "Boston United": [1.48],
                "FG Rovers": [2.74],
                "Aldershot Town": [1.02],
                "Brackley Town": [1.82],
                # "Scunthorpe Utd": [3.52], - leave this out
                "Woking": [1.46],
                "Wealdstone": [2.08],
                "Yeovil Town": [1.87],
                "Truro City": [0.87],
                "Gateshead": [1.78],
                "Carlisle United": [2.42],
                "Boreham Wood": [2.45],
                "Solihull Moors": [1.30],
                "Altrincham": [1.35],
                "Eastleigh": [2.00],
                "Rochdale": [3.33],
                "York City": [2.51],
                "Tamworth": [1.86],
                "FC Halifax Town": [2.27],
                "Hartlepool Utd": [1.81],
                "Southend United": [2.31],
                "Morecambe": [0.94],
                "Braintree Town": [1.50],
                "Sutton United": [0.89],
            }
        )
        .T.reset_index()
        .rename(columns={"index": "Team", 0: "PPI"})
        .sort_values("PPI", ascending=False)
    )

    teams = set(matches["Home"]).union(matches["Away"])

    metrics_all_teams = {
        team: [
            TeamMetrics(team_name=team, matches=matches).latest_points_performance_index
        ]
        for team in teams
    }

    computed_and_reference_ppi_df = (
        (
            pd.DataFrame(data=metrics_all_teams)
            .T.reset_index()
            .rename(columns={"index": "Team", 0: "PPI"})
        )
        .sort_values("PPI", ascending=False)
        .merge(soccerstats_reference_ppi, on="Team")
        .rename(columns={"PPI_x": "PPI_computed", "PPI_y": "PPI_reference"})
    )

    pdt.assert_series_equal(
        computed_and_reference_ppi_df["PPI_computed"],
        computed_and_reference_ppi_df["PPI_reference"],
        check_names=False,  # ignore name difference
        atol=0.05,  # absolute tolerance
        rtol=0,  # relative tolerance (optional)
    )
