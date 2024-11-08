from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from datetime import datetime

class BaseCollector(ABC):
    """Base class for data collectors"""
    
    @abstractmethod
    def get_matches(self, days: int) -> pd.DataFrame:
        """Get recent matches"""
        pass

    @abstractmethod
    def get_player_stats(self, player_id: str) -> pd.DataFrame:
        """Get player statistics"""
        pass