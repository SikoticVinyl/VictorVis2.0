import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class GridModelTrainer:
    """
    Enhanced model trainer for Grid.gg data with game-specific configurations
    """
    def __init__(self, game_type: str = 'cs2'):
        self.models = {}
        self.feature_importance = {}
        self.metrics = {}
        self.histories = {}
        self.game_type = game_type
        
    def build_neural_network(self, input_dim: int) -> Sequential:
        """
        Neural network architecture optimized for game-specific prediction
        """
        if self.game_type == 'cs2':
            # Larger network for CS2 (more data)
            model = Sequential([
                Dense(128, activation='relu', input_dim=input_dim),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dense(1)
            ])
        else:
            # Simpler network for Dota 2 (less data)
            model = Sequential([
                Dense(32, activation='relu', input_dim=input_dim),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation='relu'),
                BatchNormalization(),
                Dense(1)
            ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def build_xgboost(self) -> xgb.XGBRegressor:
        """
        XGBoost configuration based on game type
        """
        if self.game_type == 'cs2':
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                random_state=42
            )
        else:
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                random_state=42
            )
    
    def build_lightgbm(self) -> lgb.LGBMRegressor:
        """
        LightGBM configuration based on game type
        """
        if self.game_type == 'cs2':
            return lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42
            )
        else:
            return lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=15,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                min_child_samples=5,
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
        Train a single model with game-specific considerations
        """
        if model_type == 'neural_network':
            model = self.build_neural_network(X_train.shape[1])
            
            # Add learning rate reduction
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Use different batch sizes based on dataset size
            batch_size = 32 if self.game_type == 'cs2' else 16
            
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            self.histories[model_type] = history.history
            predictions = model.predict(X_test)
            
        elif model_type == 'xgboost':
            model = self.build_xgboost()
            
            if self.game_type == 'dota':
                # Use cross-validation for smaller dataset
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5,
                    scoring='neg_mean_squared_error'
                )
                self.metrics[f'{model_type}_cv'] = {
                    'mean_cv_mse': -cv_scores.mean(),
                    'std_cv_mse': cv_scores.std()
                }
            
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
            
            if self.game_type == 'dota':
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5,
                    scoring='neg_mean_squared_error'
                )
                self.metrics[f'{model_type}_cv'] = {
                    'mean_cv_mse': -cv_scores.mean(),
                    'std_cv_mse': cv_scores.std()
                }
            
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
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions))
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
        test_size: float = None
    ) -> Dict[str, Any]:
        """
        Train all models with game-specific configurations
        """
        # Use different test sizes based on game type
        if test_size is None:
            test_size = 0.2 if self.game_type == 'cs2' else 0.1
        
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