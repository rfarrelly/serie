from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Timeframe:
    start_date: date
    end_date: date

    def __post_init__(self):
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")

    def contains(self, check_date: date) -> bool:
        return self.start_date <= check_date <= self.end_date

    @property
    def days(self) -> int:
        return (self.end_date - self.start_date).days


@dataclass(frozen=True)
class Season:
    name: str
    start_year: int
    end_year: int

    def __post_init__(self):
        if self.end_year != self.start_year + 1:
            raise ValueError("Season must span exactly one year transition")
        expected_name = f"{self.start_year}-{str(self.end_year)[2:]}"
        if self.name != expected_name:
            raise ValueError(f"Season name must be {expected_name}")

    @classmethod
    def from_string(cls, season_str: str) -> "Season":
        """Create Season from string like '2023-24'"""
        try:
            start_str, end_str = season_str.split("-")
            start_year = int(start_str)
            end_year = int(f"20{end_str}")
            return cls(season_str, start_year, end_year)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid season format: {season_str}") from e
