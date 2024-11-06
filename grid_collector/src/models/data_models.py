from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class Player:
    id: str
    nickname: str
    team_id: str
    stats: Optional[Dict[str, Any]] = None

@dataclass
class Match:
    id: str
    start_time: datetime
    tournament_name: str
    team1_id: str
    team1_name: str
    team2_id: str
    team2_name: str
    team1_score: Optional[int]
    team2_score: Optional[int]
