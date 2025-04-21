import pandas as pd
from config import Leagues, DEFAULT_CONFIG, TODAY, END_DATE
from processing import LeagueProcessor


def main():
    all_bet_candidates = []

    for league in Leagues:

        print(f"Processing {league.name} ({league.value['fbref_name']})")

        processor = LeagueProcessor(league, DEFAULT_CONFIG)

        # processor.get_data()

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


if __name__ == "__main__":
    main()
