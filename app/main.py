import pandas as pd
import numpy as np
from config import Leagues, DEFAULT_CONFIG, TODAY, END_DATE, GET_DATA
from processing import LeagueProcessor, process_historical_data
from utils.team_name_dict_builder import TeamNameManagerCLI


def build_team_name_dictionary():
    data_sources = ["fbref", "fbduk"]
    csv_path = "team_name_dictionary.csv"

    # Create manager
    manager = TeamNameManagerCLI(csv_path, data_sources)

    fbduk_teams = np.unique(
        pd.concat(
            [
                pd.read_csv(str(file))[["Home", "Away"]]
                for file in DEFAULT_CONFIG.fbduk_data_dir.rglob("*.csv")
                if file.is_file()
            ]
        )
        .to_numpy()
        .flatten()
    )

    fbref_teams = np.unique(
        pd.concat(
            [
                pd.read_csv(str(file))[["Home", "Away"]]
                for file in DEFAULT_CONFIG.fbref_data_dir.rglob("*.csv")
                if file.is_file()
            ]
        )
        .to_numpy()
        .flatten()
    )

    manager.import_team_list(
        fbref_teams, "fbref", auto_match=True, auto_threshold=0.7, interactive=True
    )

    manager.import_team_list(
        fbduk_teams, "fbduk", auto_match=True, auto_threshold=0.7, interactive=True
    )

    print(f"Dictionary saved to {csv_path}")


def main():
    all_bet_candidates = []

    for league in Leagues:

        print(f"Processing {league.name} ({league.value['fbref_name']})")

        processor = LeagueProcessor(league, DEFAULT_CONFIG)

        if GET_DATA == "1":
            processor.get_fbref_data()
            processor.get_fbduk_data()

        bet_candidates = processor.generate_bet_candidates()

        if bet_candidates:
            all_bet_candidates.extend(bet_candidates)

    if all_bet_candidates:
        print(f"Getting betting candidates for the period {TODAY} to {END_DATE}")
        latest_bet_candidates_df = pd.DataFrame(all_bet_candidates).sort_values(
            by="RPI_Diff"
        )

        latest_bet_candidates_df = latest_bet_candidates_df[
            latest_bet_candidates_df["RPI_Diff"] <= DEFAULT_CONFIG.rpi_diff_threshold
        ]

        latest_bet_candidates_df.to_csv("latest_bet_candidates.csv", index=False)

    process_historical_data(DEFAULT_CONFIG).to_csv("historical_rpi.csv", index=False)


if __name__ == "__main__":
    main()
