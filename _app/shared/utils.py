from datetime import date, datetime

from .exceptions import DomainError
from .types import Season


def parse_season(season_str: str) -> Season:
    """Parse season string like '2023-24' into Season object"""
    try:
        start_str, end_str = season_str.split("-")
        start_year = int(start_str)
        end_year = int(f"20{end_str}")
        return Season(season_str, start_year, end_year)
    except (ValueError, IndexError) as e:
        raise DomainError(f"Invalid season format: {season_str}") from e


def format_date(date_value) -> str:
    """Format date consistently across the application"""
    if isinstance(date_value, str):
        return date_value
    elif isinstance(date_value, (date, datetime)):
        return date_value.strftime("%Y-%m-%d")
    else:
        return str(date_value)


def validate_probability(value: float) -> None:
    """Validate probability value is between 0 and 1"""
    if not 0 <= value <= 1:
        raise DomainError(f"Probability must be between 0 and 1, got {value}")


def validate_odds(value: float) -> None:
    """Validate odds value is greater than 1"""
    if value <= 1.0:
        raise DomainError(f"Odds must be greater than 1.0, got {value}")
