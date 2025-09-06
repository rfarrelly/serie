from dataclasses import dataclass
from enum import Enum


class League(Enum):
    PREMIER_LEAGUE = {
        "fbref_id": 9,
        "fbduk_id": "E0",
        "name": "Premier-League",
        "display_name": "Premier League",
    }
    CHAMPIONSHIP = {
        "fbref_id": 10,
        "fbduk_id": "E1",
        "name": "Championship",
        "display_name": "Championship",
    }
    BUNDESLIGA = {
        "fbref_id": 20,
        "fbduk_id": "D1",
        "name": "Bundesliga",
        "display_name": "Bundesliga",
    }
    SERIE_A = {
        "fbref_id": 11,
        "fbduk_id": "I1",
        "name": "Serie-A",
        "display_name": "Serie A",
    }
    LA_LIGA = {
        "fbref_id": 12,
        "fbduk_id": "SP1",
        "name": "La-Liga",
        "display_name": "La Liga",
    }
    LIGUE_1 = {
        "fbref_id": 13,
        "fbduk_id": "F1",
        "name": "Ligue-1",
        "display_name": "Ligue 1",
    }

    @property
    def fbref_id(self) -> int:
        return self.value["fbref_id"]

    @property
    def fbduk_id(self) -> str:
        return self.value["fbduk_id"]

    @property
    def name(self) -> str:
        return self.value["name"]

    @property
    def display_name(self) -> str:
        return self.value["display_name"]


@dataclass
class Season:
    name: str
    start_year: int
    end_year: int
