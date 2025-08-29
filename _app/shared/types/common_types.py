# shared/types/common_types.py
from decimal import Decimal
from enum import Enum
from typing import NewType

# Basic type aliases for clarity
TeamName = NewType("TeamName", str)
LeagueName = NewType("LeagueName", str)
Season = NewType("Season", str)  # e.g., "2023-2024"


class BetType(Enum):
    HOME = "Home"
    DRAW = "Draw"
    AWAY = "Away"


class ModelType(Enum):
    POISSON = "poisson"
    ZIP = "zip"
    MOV = "mov"
    ENHANCED_ZSD = "enhanced_zsd"


class BookmakerType(Enum):
    """Different bookmaker/odds source types"""

    PINNACLE = "PS"  # Sharp odds
    BET365 = "B365"  # Soft odds
    PINNACLE_CLOSING = "PSC"
    BET365_CLOSING = "B365C"
