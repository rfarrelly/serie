from enum import Enum
import os
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv, dotenv_values

load_dotenv()

env_values = dotenv_values()


class Leagues(Enum):
    EPL = {
        "fbref_id": 9,
        "fbduk_id": "E0",
        "fbref_name": "Premier-League",
        "is_extra": False,
    }
    ECH = {
        "fbref_id": 10,
        "fbduk_id": "E1",
        "fbref_name": "Championship",
        "is_extra": False,
    }
    EL1 = {
        "fbref_id": 15,
        "fbduk_id": "E2",
        "fbref_name": "League-One",
        "is_extra": False,
    }
    EL2 = {
        "fbref_id": 16,
        "fbduk_id": "E3",
        "fbref_name": "League-Two",
        "is_extra": False,
    }
    ENL = {
        "fbref_id": 34,
        "fbduk_id": "EC",
        "fbref_name": "National-League",
        "is_extra": False,
    }
    SP1 = {
        "fbref_id": 12,
        "fbduk_id": "SP1",
        "fbref_name": "La-Liga",
        "is_extra": False,
    }
    SP2 = {
        "fbref_id": 17,
        "fbduk_id": "SP2",
        "fbref_name": "Segunda-Division",
        "is_extra": False,
    }
    D1 = {
        "fbref_id": 20,
        "fbduk_id": "D1",
        "fbref_name": "Bundesliga",
        "is_extra": False,
    }
    D2 = {
        "fbref_id": 33,
        "fbduk_id": "D2",
        "fbref_name": "2-Bundesliga",
        "is_extra": False,
    }
    IT1 = {
        "fbref_id": 11,
        "fbduk_id": "I1",
        "fbref_name": "Serie-A",
        "is_extra": False,
    }
    IT2 = {
        "fbref_id": 18,
        "fbduk_id": "I2",
        "fbref_name": "Serie-B",
        "is_extra": False,
    }
    FR1 = {
        "fbref_id": 13,
        "fbduk_id": "F1",
        "fbref_name": "Ligue-1",
        "is_extra": False,
    }
    FR2 = {
        "fbref_id": 60,
        "fbduk_id": "F2",
        "fbref_name": "Ligue-2",
        "is_extra": False,
    }
    POR = {
        "fbref_id": 32,
        "fbduk_id": "P1",
        "fbref_name": "Primeira-Liga",
        "is_extra": False,
    }
    NED = {
        "fbref_id": 23,
        "fbduk_id": "N1",
        "fbref_name": "Eredivisie",
        "is_extra": False,
    }
    POL = {
        "fbref_id": 36,
        "fbduk_id": "POL",
        "fbref_name": "Ekstraklasa",
        "is_extra": True,
    }

    @property
    def fbref_id(self):
        return self.value["fbref_id"]

    @property
    def fbduk_id(self):
        return self.value["fbduk_id"]

    @property
    def fbref_name(self):
        return self.value["fbref_name"]

    @property
    def is_extra(self):
        return self.value["is_extra"]


@dataclass
class AppConfig:
    fbref_base_url: str = env_values.get("FBREF_BASE_URL")
    fbduk_base_url_main: str = env_values.get("FBDUK_BASE_URL_MAIN")
    fbduk_base_url_extra: str = env_values.get("FBDUK_BASE_URL_EXTRA")
    current_season: str = env_values.get("CURRENT_SEASON")
    ppi_diff_threshold: float = float(env_values.get("PPI_DIFF_THRESHOLD"))
    data_dir: Path = Path(env_values.get("DATA_DIR"))

    @property
    def fbref_data_dir(self):
        return self.data_dir / "FBREF"

    @property
    def fbduk_data_dir(self):
        return self.data_dir / "FBDUK"

    def get_fbref_league_dir(self, league_name):
        path = self.fbref_data_dir / league_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_fbduk_league_dir(self, league_name):
        path = self.fbduk_data_dir / league_name
        path.mkdir(parents=True, exist_ok=True)
        return path


TODAY = datetime.now().date()
TIME_DELTA = int(env_values.get("TIME_DELTA"))
END_DATE = TODAY + timedelta(days=TIME_DELTA)

GET_DATA = env_values.get("GET_DATA")

# Default app configuration
DEFAULT_CONFIG = AppConfig()
