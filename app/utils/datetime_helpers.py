import pandas as pd
from datetime import datetime


def filter_date_range(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    filtered_df = df[
        (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    ]
    return filtered_df
