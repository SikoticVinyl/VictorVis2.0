import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

class GridDataPreparation:
    """
    Prepares CS2 and Dota 2 data for model training with proper edge case handling
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
        
        # Handle zero values before division
        X['avg_deaths_per_round'] = X['avg_deaths_per_round'].replace(0, np.nan)
        
        # Create engineered features with safe division
        X['kd_ratio'] = X['avg_kills_per_round'] / X['avg_deaths_per_round']
        X['kd_ratio'] = X['kd_ratio'].fillna(X['avg_kills_per_round'])  # If deaths is 0, use kills
        
        # Create objective score with safe division
        X['objective_score'] = (
            X['defuse_with_kit_total'] + 
            X['defuse_without_kit_total'] * 1.5 +
            X['plant_bomb_total'] +
            X['explode_bomb_total'] * 1.5
        )
        # Avoid division by zero for games_played
        X['objective_score'] = np.where(
            X['games_played'] > 0,
            X['objective_score'] / X['games_played'],
            0
        )
        
        # Log transform games_played (add 1 to handle zeros)
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
        
        # Handle infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Scale features
        scaled_features = self.scalers['cs2'].fit_transform(X)
        X_scaled = pd.DataFrame(scaled_features, columns=X.columns)
        
        return X_scaled, y
    
    def prepare_dota_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare Dota 2 data for modeling with safe calculations
        """
        # Select relevant features
        features = [
            'games_played',
            'avg_kills',
            'avg_deaths',
            'first_kills_percentage',
            'experience_avg'
        ]
        
        # Create feature DataFrame
        X = df[features].copy()
        
        # Handle zero values before division
        X['avg_deaths'] = X['avg_deaths'].replace(0, np.nan)
        X['games_played'] = X['games_played'].replace(0, np.nan)
        
        # Create target (experience efficiency) with safe division
        y = np.where(
            df['games_played'] > 0,
            df['experience_avg'] / df['games_played'],
            df['experience_avg']  # If no games, use raw experience
        )
        
        # Create kd_ratio safely
        X['kd_ratio'] = X['avg_kills'] / X['avg_deaths']
        X['kd_ratio'] = X['kd_ratio'].fillna(X['avg_kills'])  # If deaths is 0, use kills
        
        # Log transform games_played
        X['games_played'] = np.log1p(X['games_played'])
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Handle infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.mean())
        
        # Scale features
        scaled_features = self.scalers['dota'].fit_transform(X)
        X_scaled = pd.DataFrame(scaled_features, columns=X.columns)
        
        return X_scaled, pd.Series(y)
    
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