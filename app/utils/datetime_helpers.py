import pandas as pd
from config import TODAY, END_DATE


def filter_date_range(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    filtered_df = df[(df["Date"].dt.date >= TODAY) & (df["Date"].dt.date <= END_DATE)]
    return filtered_df
