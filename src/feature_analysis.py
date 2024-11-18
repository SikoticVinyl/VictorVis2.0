import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import List, Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Game-specific metric definitions
CS2_PERFORMANCE_METRICS = [
    'games_played',
    'series_played',
    'avg_kills',
    'avg_deaths',
    'best_kills',
    'avg_kills_per_round',
    'avg_deaths_per_round',
    'first_kills_total',
    'first_kills_percentage',
    'damage_avg',
    'damage_max',
    'avg_win_rate'
]

CS2_UTILITY_METRICS = [
    'games_played',
    'total_rounds',
    'defuse_with_kit_total',
    'defuse_without_kit_total',
    'explode_bomb_total',
    'plant_bomb_total',
    'total_segment_kills',
    'total_segment_deaths'
]

DOTA2_PERFORMANCE_METRICS = [
    'games_played',
    'series_played',
    'avg_kills',
    'avg_deaths',
    'best_kills',
    'first_kills_total',
    'first_kills_percentage',
    'experience_avg',
    'experience_max',
    'experience_total'
]

def calculate_experience_weight(games_played: pd.Series) -> pd.Series:
    """
    Calculate experience weight based on games played.
    
    Args:
        games_played (pd.Series): Series containing number of games played
        
    Returns:
        pd.Series: Experience weights
    """
    min_games = 5
    log_games = np.log1p(games_played)
    weight = np.where(games_played >= min_games,
                     log_games / log_games.median(),
                     0.5)
    return pd.Series(weight, index=games_played.index)

def calculate_performance_score(df: pd.DataFrame, game_type: str = 'cs2') -> pd.Series:
    """
    Calculate a performance score based on game-specific metrics.
    """
    if game_type == 'cs2':
        score = (
            df['avg_kills_per_round'] * 100 +
            df['first_kills_percentage'] * 0.5 -
            df['avg_deaths_per_round'] * 50 +
            df['avg_win_rate'] +
            (df['damage_avg'] / 1000) * 20 +
            (df['best_kills'] / df['games_played']) * 10
        )
    else:  # dota2
        score = (
            df['avg_kills'] * 100 -
            df['avg_deaths'] * 50 +
            df['experience_avg'] / 1000 +
            (df['experience_max'] / df['experience_avg']) * 20 +
            (df['best_kills'] / df['games_played']) * 10
        )
    
    # Weight by games played for both games
    games_weight = np.log1p(df['games_played']) / np.log1p(df['games_played'].median())
    weighted_score = score * games_weight
    
    return weighted_score

def analyze_game_metrics(df: pd.DataFrame, 
                        game_type: str,
                        metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Comprehensive analysis of game metrics including experience-based insights.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        game_type (str): Either 'cs2' or 'dota2'
        metrics (List[str]): List of metrics to analyze
        
    Returns:
        Tuple containing:
        - Statistics DataFrame
        - Correlation matrix
        - Experience-based analysis DataFrame
    """
    stats_df = df[metrics].describe()
    corr_matrix = df[metrics].corr()
    exp_analysis = analyze_player_experience(df, game_type)
    
    return stats_df, corr_matrix, exp_analysis

def analyze_player_experience(df: pd.DataFrame, game_type: str) -> pd.DataFrame:
    """
    Detailed analysis of player experience levels and consistency.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        game_type (str): Either 'cs2' or 'dota2'
        
    Returns:
        pd.DataFrame: Experience analysis results
    """
    df = df.copy()
    
    # Define experience levels
    df['experience_level'] = pd.qcut(df['games_played'], q=4, 
                                   labels=['Rookie', 'Developing', 'Experienced', 'Veteran'])
    
    # Calculate consistency metrics
    df['consistency_score'] = df.groupby('experience_level')['avg_kills'].transform(
        lambda x: 1 - x.std() / x.mean()
    )
    
    # Calculate peak performance
    if game_type == 'cs2':
        df['peak_performance'] = df['best_kills'] / df['avg_kills']
    else:
        df['peak_performance'] = df['experience_max'] / df['experience_avg']
    
    # Aggregate metrics by experience level
    exp_analysis = df.groupby('experience_level').agg({
        'games_played': ['count', 'mean'],
        'avg_kills': ['mean', 'std'],
        'avg_deaths': ['mean', 'std'],
        'consistency_score': 'mean',
        'peak_performance': 'mean'
    }).round(2)
    
    return exp_analysis

def plot_metric_distributions(df: pd.DataFrame, 
                            metrics: List[str],
                            game_name: str) -> None:
    """
    Plot distributions for game metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        metrics (List[str]): List of metrics to plot
        game_name (str): Name of the game for plot titles
    """
    num_metrics = len(metrics)
    num_rows = (num_metrics + 2) // 3
    
    plt.figure(figsize=(15, 5 * num_rows))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(num_rows, 3, i)
        sns.histplot(data=df, x=metric, kde=True)
        plt.title(f'{metric} Distribution')
        plt.xticks(rotation=45)
    plt.suptitle(f'{game_name} Metric Distributions')
    plt.tight_layout()
    plt.show()

def plot_performance_correlations(df: pd.DataFrame, 
                                metrics: List[str],
                                game_name: str) -> None:
    """
    Plot correlation heatmap for performance metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        metrics (List[str]): List of metrics to analyze
        game_name (str): Name of the game for plot titles
    """
    corr_matrix = df[metrics].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'{game_name} Performance Metric Correlations')
    plt.tight_layout()
    plt.show()

def analyze_player_roles(df: pd.DataFrame, game_type: str) -> pd.DataFrame:
    """
    Analyze player roles based on game-specific metrics.
    """
    df = df.copy()
    
    if game_type == 'cs2':
        # Entry fragger rating
        df['entry_rating'] = (
            df['first_kills_percentage'] * 
            (df['avg_kills_per_round'] / df['avg_deaths_per_round']) *
            (df['damage_avg'] / df['damage_avg'].mean())
        )
        
        # Support rating
        df['support_rating'] = (
            (df['defuse_with_kit_total'] + df['defuse_without_kit_total']) / 
            df['games_played']
        )
        
        # Objective rating
        df['objective_rating'] = (
            (df['plant_bomb_total'] + df['explode_bomb_total']) / 
            df['games_played'] *
            (df['avg_win_rate'] / 50)
        )
        
    else:  # dota2
        # Combat efficiency
        df['combat_rating'] = (
            df['avg_kills'] * 
            (1 - df['avg_deaths'] / 10) *
            (df['best_kills'] / df['avg_kills'])
        )
        
        # Resource efficiency
        df['farm_rating'] = (
            (df['experience_avg'] / df['games_played']) *
            (df['experience_max'] / df['experience_total'])
        )
        
        # Early game impact
        df['early_game_rating'] = (
            df['first_kills_percentage'] *
            (df['first_kills_total'] / df['games_played'])
        )
    
    return df

def generate_game_report(df: pd.DataFrame,
                        game_type: str,
                        include_plots: bool = True) -> Dict[str, Any]:
    """
    Generate game-specific analysis report.
    """
    df = df.copy()
    metrics = CS2_PERFORMANCE_METRICS if game_type == 'cs2' else DOTA2_PERFORMANCE_METRICS
    game_name = "Counter-Strike 2" if game_type == 'cs2' else "Dota 2"
    
    # Calculate performance scores
    df['performance_score'] = calculate_performance_score(df, game_type)
    
    # Analyze metrics
    stats_df = df[metrics].describe()
    corr_matrix = df[metrics].corr()
    
    # Add role analysis
    df = analyze_player_roles(df, game_type)
    
    # Generate plots if requested
    if include_plots:
        plot_metric_distributions(df, metrics, game_name)
        plot_performance_correlations(df, metrics, game_name)
    
    # Compile report
    report = {
        'game_name': game_name,
        'metrics_analyzed': metrics,
        'basic_stats': stats_df,
        'correlations': corr_matrix,
        'top_performers': df.nlargest(10, 'performance_score')[
            ['nickname', 'performance_score', 'games_played'] + 
            [col for col in metrics if col not in ['games_played']]
        ],
        'performance_distribution': df['performance_score'].describe(),
        'role_distribution': df[[col for col in df.columns if col.endswith('_rating')]].describe()
    }
    
    return report

def print_game_report(report: Dict[str, Any]) -> None:
    """
    Print formatted game analysis report.
    """
    print(f"\n{'='*20} {report['game_name']} Analysis {'='*20}")
    
    print("\nMetrics Analyzed:")
    print(report['metrics_analyzed'])
    
    print("\nBasic Statistics:")
    print(report['basic_stats'])
    
    print("\nTop Performers:")
    print(report['top_performers'])
    
    print("\nPerformance Score Distribution:")
    print(report['performance_distribution'])
    
    print("\nRole Distribution:")
    print(report['role_distribution'])

def normalize_game_features(df: pd.DataFrame,
                          game_type: str,
                          exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize game-specific features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        game_type (str): Either 'cs2' or 'dota2'
        exclude_cols (List[str], optional): Columns to exclude from normalization
        
    Returns:
        Tuple containing:
        - Normalized DataFrame
        - Fitted StandardScaler object
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Add default exclusions
    exclude_cols.extend(['nickname', 'team_id', 'team_name'])
    
    # Select appropriate metrics
    metrics = CS2_PERFORMANCE_METRICS if game_type == 'cs2' else DOTA2_PERFORMANCE_METRICS
    cols_to_normalize = [col for col in metrics if col not in exclude_cols]
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Create copy of DataFrame
    normalized_df = df.copy()
    
    # Normalize selected columns
    normalized_df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    return normalized_df, scaler