import pandas as pd
from config import Leagues, DEFAULT_CONFIG, TODAY, DAYS_AHEAD
from processing import LeagueProcessor


def main():
    all_candidates = []

    for league in Leagues:

        processor = LeagueProcessor(league, DEFAULT_CONFIG)

        print(f"Processing {league.name} ({league.value['fbref_name']})")

        processor.get_data()

        league_candidates = processor.compute_league_rpi()

        if league_candidates:
            all_candidates.extend(league_candidates)

    if all_candidates:
        print(f"Getting betting candidates for the period {TODAY} to {DAYS_AHEAD}")
        candidates_df = pd.DataFrame(all_candidates).sort_values(by="RPI_Diff")
        candidates_df = candidates_df[
            candidates_df["RPI_Diff"] <= DEFAULT_CONFIG.rpi_diff_threshold
        ]
        candidates_df.to_csv("candidates.csv", index=False)
        print("Saved sorted candidate matches to candidates.csv")


if __name__ == "__main__":
    main()
