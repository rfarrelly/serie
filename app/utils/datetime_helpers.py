from datetime import date, datetime
from typing import Union

import pandas as pd


def filter_date_range(
    df: pd.DataFrame,
    start_date: Union[datetime, date, str],
    end_date: Union[datetime, date, str],
) -> pd.DataFrame:
    """Filter dataframe by date range with proper date handling."""
    if df.empty:
        return df

    df = df.copy()

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove rows with invalid dates
    df = df[df["Date"].notna()]

    if len(df) == 0:
        print("No valid dates found in dataframe")
        return df

    # Convert start and end dates to consistent format
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()
    elif not isinstance(start_date, date):
        start_date = pd.to_datetime(start_date).date()

    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()
    elif not isinstance(end_date, date):
        end_date = pd.to_datetime(end_date).date()

    # Extract dates for comparison
    df_dates = df["Date"].dt.date

    # Apply filter
    mask = (df_dates >= start_date) & (df_dates <= end_date)
    filtered_df = df[mask]

    print(f"Date filter: {start_date} to {end_date}")
    print(f"Available dates: {df_dates.min()} to {df_dates.max()}")
    print(f"Matches in range: {len(filtered_df)}/{len(df)}")

    return filtered_df


def format_date(df: pd.DataFrame) -> pd.DataFrame:
    """Format date column to standard format."""
    df = df.copy()
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        # Convert back to string in YYYY-MM-DD format
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Warning: Error formatting dates: {e}")
        # Try alternative parsing
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return df


def normalize_date_format(date_input: Union[str, datetime, date]) -> str:
    """Normalize various date inputs to YYYY-MM-DD string format."""
    if isinstance(date_input, str):
        try:
            return pd.to_datetime(date_input).strftime("%Y-%m-%d")
        except:
            return date_input
    elif isinstance(date_input, (datetime, date)):
        return date_input.strftime("%Y-%m-%d")
    else:
        try:
            return pd.to_datetime(date_input).strftime("%Y-%m-%d")
        except:
            return str(date_input)
