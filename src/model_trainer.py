import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

class GridModelTrainer:
    """
    Base model trainer for Grid.gg data with simple configurations
    to allow for later optimization demonstrations
    """
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.metrics = {}
        self.histories = {}  # Store training history for NN
        
    def build_neural_network(self, input_dim: int) -> Sequential:
        """
        Basic neural network architecture - can be optimized later
        """
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_xgboost(self) -> xgb.XGBRegressor:
        """
        Basic XGBoost with starter parameters
        """
        return xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    
    def build_lightgbm(self) -> lgb.LGBMRegressor:
        """
        Basic LightGBM with starter parameters
        """
        return lgb.LGBMRegressor(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
            random_state=42
        )
    
    def train_model(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray, 
        y_train: np.ndarray, 
        y_test: np.ndarray,
        model_type: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Train a single model and store basic metrics
        """
        if model_type == 'neural_network':
            model = self.build_neural_network(X_train.shape[1])
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.histories[model_type] = history.history
            predictions = model.predict(X_test)
            
        elif model_type == 'xgboost':
            model = self.build_xgboost()
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            predictions = model.predict(X_test)
            self.feature_importance[model_type] = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        else:  # lightgbm
            model = self.build_lightgbm()
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            predictions = model.predict(X_test)
            self.feature_importance[model_type] = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Calculate basic metrics
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        self.models[model_type] = model
        self.metrics[model_type] = metrics
        
        return {
            'model': model,
            'predictions': predictions,
            'metrics': metrics
        }
    
    def train_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train all three models with basic configurations
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        results = {}
        for model_type in ['neural_network', 'xgboost', 'lightgbm']:
            results[model_type] = self.train_model(
                X_train, X_test, y_train, y_test, model_type, feature_names
            )
        
        return {
            'models': self.models,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'histories': self.histories,
            'test_data': (X_test, y_test),
            'train_data': (X_train, y_train)
        }