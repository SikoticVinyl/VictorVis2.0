import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

class GridDataPreparation:
    """
    Prepares CS2 and Dota 2 data for model training
    """
    def __init__(self):
        self.scalers = {
            'cs2': StandardScaler(),
            'dota': StandardScaler()
        }
        
    def prepare_cs2_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare CS2 data for modeling
        """
        # Select relevant features based on analysis
        features = [
            'games_played',
            'avg_kills_per_round',
            'avg_deaths_per_round',
            'first_kills_percentage',
            'damage_avg',
            'defuse_with_kit_total',
            'defuse_without_kit_total',
            'explode_bomb_total',
            'plant_bomb_total'
        ]
        
        # Target variable
        target = 'avg_win_rate'
        
        # Create feature DataFrame
        X = df[features].copy()
        y = df[target].copy()
        
        # Create engineered features
        X['kd_ratio'] = X['avg_kills_per_round'] / X['avg_deaths_per_round']
        X['objective_score'] = (
            X['defuse_with_kit_total'] + 
            X['defuse_without_kit_total'] * 1.5 +
            X['plant_bomb_total'] +
            X['explode_bomb_total'] * 1.5
        ) / X['games_played']
        
        # Log transform games_played
        X['games_played'] = np.log1p(X['games_played'])
        
        # Drop original objective columns after creating score
        X = X.drop([
            'defuse_with_kit_total',
            'defuse_without_kit_total',
            'explode_bomb_total',
            'plant_bomb_total'
        ], axis=1)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaled_features = self.scalers['cs2'].fit_transform(X)
        X_scaled = pd.DataFrame(scaled_features, columns=X.columns)
        
        return X_scaled, y
    
    def prepare_dota_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare Dota 2 data for modeling
        """
        # Select relevant features
        features = [
            'games_played',
            'avg_kills',
            'avg_deaths',
            'first_kills_percentage',
            'experience_avg'
        ]
        
        # Use experience efficiency as target
        X = df[features].copy()
        y = df['experience_avg'] / df['games_played']  # Experience efficiency as target
        
        # Create engineered features
        X['kd_ratio'] = X['avg_kills'] / X['avg_deaths']
        
        # Log transform games_played
        X['games_played'] = np.log1p(X['games_played'])
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaled_features = self.scalers['dota'].fit_transform(X)
        X_scaled = pd.DataFrame(scaled_features, columns=X.columns)
        
        return X_scaled, y
    
    def prepare_data_for_training(self, cs2_df: pd.DataFrame, dota_df: pd.DataFrame) -> Dict:
        """
        Prepare both games' data for training
        """
        # Prepare each dataset
        cs2_X, cs2_y = self.prepare_cs2_data(cs2_df)
        dota_X, dota_y = self.prepare_dota_data(dota_df)
        
        return {
            'cs2': {
                'X': cs2_X,
                'y': cs2_y,
                'feature_names': cs2_X.columns.tolist()
            },
            'dota': {
                'X': dota_X,
                'y': dota_y,
                'feature_names': dota_X.columns.tolist()
            }
        }