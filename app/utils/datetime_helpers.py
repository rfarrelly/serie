import pandas as pd
from config import TODAY, DAYS_AHEAD


def filter_date_range(df, date_column):
    end_date = DAYS_AHEAD

    df[date_column] = pd.to_datetime(df[date_column])

    filtered_df = df[
        (df[date_column].dt.date >= TODAY) & (df[date_column].dt.date <= end_date)
    ]

    return filtered_df
