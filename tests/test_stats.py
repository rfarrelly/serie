import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from app.common.stats import TeamStats


class TestTeamStats(unittest.TestCase):
    def setUp(self):
        # Set up a sample DataFrame for testing
        data = {
            "Wk": [1.0, 1.0, 2.0, 2.0],
            "Day": ["Fri", "Sat", "Sat", "Sun"],
            "Date": ["2024-08-16", "2024-08-17", "2024-08-24", "2024-08-25"],
            "Time": ["20:00", "12:30", "15:00", "16:30"],
            "HomeTeam": ["Manchester Utd", "Ipswich Town", "Brighton", "Liverpool"],
            "AwayTeam": ["Fulham", "Liverpool", "Manchester Utd", "Brentford"],
            "FTHG": [1, 0, 2, 2],
            "FTAG": [0, 2, 1, 0],
        }
        self.df = pd.DataFrame(data)

    def test_init__success(self):
        team = "Manchester Utd"
        stats = TeamStats(team, self.df)

        # Check if the DataFrame is filtered correctly
        data = {
            "Wk": [1.0, 2.0],
            "Day": ["Fri", "Sat"],
            "Date": ["2024-08-16", "2024-08-24"],
            "Time": ["20:00", "15:00"],
            "HomeTeam": ["Manchester Utd", "Brighton"],
            "AwayTeam": ["Fulham", "Manchester Utd"],
            "FTHG": [1, 2],
            "FTAG": [0, 1],
            "HomePoints": [3, None],
            "AwayPoints": [None, 0],
            "PPG": [3, 1.5],
            "Team": ["Manchester Utd", "Manchester Utd"],
            "HomeOpps": [None, "Brighton"],
            "AwayOpps": ["Fulham", None],
        }
        expected_df = pd.DataFrame(data)

        assert_frame_equal(stats.stats_df, expected_df)

    def test__compute_home_and_away_points_success(self):
        team = "Manchester Utd"
        stats = TeamStats(team, self.df)

        # Compute home and away points manually
        expected_df = stats.stats_df.copy()
        expected_df["HomePoints"] = expected_df.apply(
            lambda x: (
                (3 if x["FTHG"] > x["FTAG"] else 1 if x["FTHG"] == x["FTAG"] else 0)
                if x["HomeTeam"] == team
                else None
            ),
            axis="columns",
        )
        expected_df["AwayPoints"] = expected_df.apply(
            lambda x: (
                (3 if x["FTHG"] < x["FTAG"] else 1 if x["FTHG"] == x["FTAG"] else 0)
                if x["AwayTeam"] == team
                else None
            ),
            axis="columns",
        )

        stats._setup()
        assert_frame_equal(stats.stats_df, expected_df)

    def test__setup__success(self):
        team = "Manchester Utd"
        stats = TeamStats(team, self.df)

        computed_ppg = stats.stats_df["PPG"].iloc[-1]

        # Expected points: 3 (vs Fulham) + 0 (vs Brighton)
        expected_ppg = (3 + 0) / 2  # Total points divided by number of games
        self.assertAlmostEqual(computed_ppg, expected_ppg, places=5)

    def test_get_stats_df__success(self):
        team = "Liverpool"
        stats = TeamStats(team, self.df)
        filtered_df = stats.stats_df

        # Ensure the DataFrame only contains rows related to the specified team
        self.assertTrue(
            all((filtered_df["HomeTeam"] == team) | (filtered_df["AwayTeam"] == team))
        )


if __name__ == "__main__":
    unittest.main()
