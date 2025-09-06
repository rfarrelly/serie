from dataclasses import dataclass


@dataclass
class Team:
    name: str
    league: str
    season: str

    def __str__(self) -> str:
        return self.name


@dataclass
class TeamRatings:
    team_name: str
    attack_rating: float
    defense_rating: float
    net_rating: float

    @property
    def strength(self) -> float:
        return self.attack_rating - self.defense_rating
