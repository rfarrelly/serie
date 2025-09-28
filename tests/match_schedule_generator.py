# schedule.py
import random
from datetime import datetime, timedelta
from itertools import combinations

import pandas as pd


def generate_schedule(teams, start_date="2025-08-09", seed=42):
    pairs = list(combinations(teams, 2))
    rounds, used = [], set()

    for _ in range(len(teams) - 1):  # single round robin
        round_matches, seen = [], set()
        for a, b in pairs:
            if a not in seen and b not in seen and (a, b) not in used:
                round_matches.append((a, b))
                seen.update([a, b])
                used.add((a, b))
        rounds.append(round_matches)

    # add reverse fixtures
    schedule = rounds + [[(b, a) for a, b in r] for r in rounds]

    # flatten into DataFrame
    df = pd.DataFrame(
        [(wk + 1, h, a) for wk, r in enumerate(schedule) for h, a in r],
        columns=["Wk", "Home", "Away"],
    )

    # add date column
    start_date = datetime.fromisoformat(start_date)
    df["Date"] = df["Wk"].apply(
        lambda w: (start_date + timedelta(weeks=w - 1)).strftime("%Y-%m-%d")
    )

    # add goals
    random.seed(seed)
    df["FTHG"] = [random.randint(0, 5) for _ in range(len(df))]
    df["FTAG"] = [random.randint(0, 5) for _ in range(len(df))]

    return df
