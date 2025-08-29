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
