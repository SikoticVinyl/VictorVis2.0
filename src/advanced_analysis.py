import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, Tuple, List

class GameMetricsAnalyzer:
    """Base class for game-specific analysis"""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def analyze_consistency(self) -> pd.DataFrame:
        raise NotImplementedError
        
    def analyze_specialization(self) -> pd.DataFrame:
        raise NotImplementedError
        
    def analyze_impact(self) -> pd.DataFrame:
        raise NotImplementedError

class CS2Analyzer(GameMetricsAnalyzer):
    """CS2-specific analysis implementation"""
    def analyze_consistency(self) -> pd.DataFrame:
        # CS2-specific consistency metrics
        self.df['kills_consistency'] = self.df['avg_kills'] / self.df['best_kills']
        self.df['damage_consistency'] = self.df['damage_avg'] / self.df['damage_max']
        self.df['round_consistency'] = 1 - (
            self.df['avg_deaths_per_round'] / self.df['avg_kills_per_round']
        )
        
        return self.df
    
    def analyze_specialization(self) -> pd.DataFrame:
        # CS2-specific role analysis
        self.df['entry_rating'] = (
            self.df['first_kills_total'] / self.df['games_played'] * 
            self.df['first_kills_percentage'] / 100
        )
        
        self.df['survival_rating'] = (
            (1 - self.df['avg_deaths_per_round']) * 
            self.df['avg_win_rate'] / 100
        )
        
        self.df['support_rating'] = (
            (self.df['damage_avg'] / self.df['avg_kills']) * 
            (self.df['avg_win_rate'] / 50)
        )
        
        return self.df
    
    def analyze_impact(self) -> pd.DataFrame:
        # CS2-specific impact metrics
        self.df['round_impact'] = (
            self.df['avg_kills_per_round'] * 
            (1 - self.df['avg_deaths_per_round']) * 
            (self.df['damage_avg'] / 1000)
        )
        
        self.df['opening_impact'] = (
            self.df['first_kills_percentage'] * 
            self.df['avg_win_rate'] / 100
        )
        
        return self.df

class Dota2Analyzer(GameMetricsAnalyzer):
    """Dota2-specific analysis implementation"""
    def analyze_consistency(self) -> pd.DataFrame:
        # Dota2-specific consistency metrics
        self.df['kills_consistency'] = self.df['avg_kills'] / self.df['best_kills']
        self.df['exp_consistency'] = self.df['experience_avg'] / self.df['experience_max']
        
        return self.df
    
    def analyze_specialization(self) -> pd.DataFrame:
        # Dota2-specific role analysis
        self.df['carry_rating'] = (
            self.df['experience_avg'] * 
            (self.df['avg_kills'] / (self.df['avg_deaths'] + 1))
        )
        
        self.df['early_game_rating'] = (
            self.df['first_kills_total'] / self.df['games_played'] * 
            self.df['first_kills_percentage'] / 100
        )
        
        return self.df
    
    def analyze_impact(self) -> pd.DataFrame:
        # Dota2-specific impact metrics
        self.df['game_impact'] = (
            self.df['avg_kills'] * 
            (self.df['experience_avg'] / self.df['experience_avg'].mean()) * 
            (1 / (self.df['avg_deaths'] + 1))
        )
        
        return self.df

def perform_clustering(df: pd.DataFrame, 
                      game_type: str,
                      n_clusters: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform clustering analysis based on game-specific metrics
    """
    if game_type == 'cs2':
        features = [
            'avg_kills_per_round', 'avg_deaths_per_round',
            'first_kills_percentage', 'damage_avg',
            'avg_win_rate'
        ]
    else:  # dota2
        features = [
            'avg_kills', 'avg_deaths',
            'experience_avg', 'first_kills_percentage'
        ]
    
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Apply PCA
    pca = PCA(n_components=min(3, len(features)))
    pca_result = pca.fit_transform(scaled_features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['player_cluster'] = kmeans.fit_predict(pca_result)
    
    # Calculate cluster profiles
    cluster_profiles = df.groupby('player_cluster')[features].mean()
    
    return df, cluster_profiles

def generate_game_report(df: pd.DataFrame, game_type: str) -> Dict:
    """
    Generate comprehensive analysis report for specified game type
    """
    # Initialize appropriate analyzer
    analyzer = CS2Analyzer(df) if game_type == 'cs2' else Dota2Analyzer(df)
    
    # Perform analyses
    df = analyzer.analyze_consistency()
    df = analyzer.analyze_specialization()
    df = analyzer.analyze_impact()
    
    # Perform clustering
    df, cluster_profiles = perform_clustering(df, game_type)
    
    # Generate report
    report = {
        'game_type': game_type,
        'player_count': len(df),
        'experience_distribution': df['games_played'].describe(),
        'consistency_metrics': df[[col for col in df.columns if 'consistency' in col]].describe(),
        'specialization_metrics': df[[col for col in df.columns if 'rating' in col]].describe(),
        'impact_metrics': df[[col for col in df.columns if 'impact' in col]].describe(),
        'cluster_profiles': cluster_profiles,
        'top_performers': df.nlargest(10, 'games_played')[
            ['nickname'] + 
            [col for col in df.columns if any(x in col for x in ['rating', 'impact', 'consistency'])]
        ]
    }
    
    return report

def print_game_report(report: Dict) -> None:
    """Print formatted analysis report"""
    print(f"\n=== {report['game_type'].upper()} Comprehensive Analysis ===")
    
    print(f"\nPlayer Base: {report['player_count']} players")
    
    print("\nExperience Distribution:")
    print(report['experience_distribution'])
    
    print("\nConsistency Metrics:")
    print(report['consistency_metrics'])
    
    print("\nSpecialization Metrics:")
    print(report['specialization_metrics'])
    
    print("\nImpact Metrics:")
    print(report['impact_metrics'])
    
    print("\nPlayer Clusters:")
    print(report['cluster_profiles'])
    
    print("\nTop Performers:")
    print(report['top_performers'])