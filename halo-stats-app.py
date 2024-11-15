# app.py

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class HaloStatsAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('HALO_API_KEY')
        self.base_url = "https://www.haloapi.com/stats/h5"
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        self.model = None
        self.scaler = StandardScaler()
    
    def fetch_match_data(self, player_gamertag, count=25):
        """Fetch recent matches for a player"""
        endpoint = f"{self.base_url}/players/{player_gamertag}/matches"
        params = {
            'count': count
        }
        
        response = requests.get(endpoint, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed: {response.status_code}")
    
    def process_match_data(self, matches_data):
        """Process raw match data into features"""
        processed_data = []
        
        for match in matches_data['Results']:
            player_stats = match['PlayerStats'][0]  # Get stats for the queried player
            
            match_data = {
                'kills': player_stats['TotalKills'],
                'deaths': player_stats['TotalDeaths'],
                'assists': player_stats['TotalAssists'],
                'shots_fired': player_stats['TotalShotsFired'],
                'shots_hit': player_stats['TotalShotsLanded'],
                'damage_dealt': player_stats['TotalWeaponDamage'],
                'win': 1 if player_stats['Result'] == 3 else 0  # 3 indicates victory
            }
            
            processed_data.append(match_data)
        
        return pd.DataFrame(processed_data)
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        df['kd_ratio'] = df['kills'] / df['deaths'].replace(0, 1)
        df['accuracy'] = df['shots_hit'] / df['shots_fired']
        df['avg_damage_per_kill'] = df['damage_dealt'] / df['kills'].replace(0, 1)
        
        return df
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the prediction model"""
        # Create model architecture
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC', 'Precision', 'Recall']
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        self.model = model
        return history
    
    def analyze_player(self, gamertag, fetch_count=25):
        """Complete analysis pipeline for a player"""
        # Fetch and process data
        raw_data = self.fetch_match_data(gamertag, fetch_count)
        df = self.process_match_data(raw_data)
        df = self.prepare_features(df)
        
        # Prepare features
        X = df.drop('win', axis=1)
        y = df['win']
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        history = self.train_model(X_train_scaled, y_train, X_test_scaled, y_test)
        
        return history, X_test_scaled, y_test
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss
        axes[0,0].plot(history.history['loss'], label='Training Loss')
        axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0,0].set_title('Model Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        
        # Accuracy
        axes[0,1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0,1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0,1].set_title('Model Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        
        plt.tight_layout()
        plt.show()

