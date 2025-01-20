import unittest
from unittest.mock import patch
import os
from io import StringIO
from app.common.ingestion import get_fbref_data
from app.common.config import FbrefLeagueName


class TestGetFbrefData(unittest.TestCase):
    @patch("app.common.ingestion.requests.get")
    @patch("app.common.ingestion.pd.DataFrame.to_csv")
    def test_get_fbref_data__success(self, mock_to_csv, mock_requests):
        with open(
            "tests/input_data/fbref_scores_and_fixtures_response_data.txt", "r"
        ) as file:
            mock_response_content = file.read()

        mock_requests.return_value.content = StringIO(mock_response_content)

        # Directory and file names
        test_dir = "app/DATA/FBREF"
        test_league = FbrefLeagueName.EPL
        test_season = "2024-2025"
        expected_filename = f"{test_league.value}_{test_season}.csv".replace("-", "_")

        # Call the function
        get_fbref_data("https://test.com", test_league, test_season, test_dir)

        # Check that to_csv was called with the correct path and content
        mock_to_csv.assert_called_once()
        csv_path = os.path.join(test_dir, expected_filename)
        self.assertEqual(mock_to_csv.call_args[0][0], csv_path)

        # Verify the saved DataFrame
        saved_df = mock_to_csv.call_args[1]["index"]
        self.assertFalse(saved_df)
