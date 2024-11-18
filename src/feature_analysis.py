import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def analyze_numeric_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Analyze numeric features in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        Tuple containing:
        - Basic statistics DataFrame
        - Correlation matrix
        - Skewness Series
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    stats_df = df[numeric_cols].describe()
    corr_matrix = df[numeric_cols].corr()
    skewness = df[numeric_cols].skew()
    
    return stats_df, corr_matrix, skewness

def plot_feature_distributions(df: pd.DataFrame, numeric_cols: List[str] = None) -> None:
    """
    Plot distributions for numeric features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_cols (List[str], optional): List of numeric columns to plot. 
                                          If None, all numeric columns are used.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    num_features = len(numeric_cols)
    num_rows = (num_features + 2) // 3  # Ceiling division
    
    plt.figure(figsize=(15, 5 * num_rows))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(num_rows, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} Distribution')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Feature Correlations") -> None:
    """
    Plot correlation heatmap.
    
    Args:
        corr_matrix (pd.DataFrame): Correlation matrix
        title (str): Title for the heatmap
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def identify_key_features(corr_matrix: pd.DataFrame, 
                         target_cols: List[str], 
                         threshold: float = 0.5) -> Dict[str, pd.Series]:
    """
    Identify features with strong correlations to target variables.
    
    Args:
        corr_matrix (pd.DataFrame): Correlation matrix
        target_cols (List[str]): List of target column names
        threshold (float): Correlation threshold for feature importance
        
    Returns:
        Dict mapping target columns to their strongly correlated features
    """
    important_correlations = {}
    for target in target_cols:
        strong_corrs = corr_matrix[target][
            (abs(corr_matrix[target]) > threshold) & 
            (corr_matrix.index != target)
        ].sort_values(ascending=False)
        important_correlations[target] = strong_corrs
    return important_correlations

def analyze_categorical_features(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Analyze categorical features in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        Dict containing value counts for each categorical column
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_stats = {}
    
    for col in categorical_cols:
        categorical_stats[col] = df[col].value_counts().to_dict()
    
    return categorical_stats

def generate_feature_report(df: pd.DataFrame, 
                          game_name: str,
                          target_cols: List[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive feature analysis report.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        game_name (str): Name of the game being analyzed
        target_cols (List[str], optional): List of target columns for correlation analysis
        
    Returns:
        Dict containing analysis results
    """
    # Basic dataset info
    report = {
        'game_name': game_name,
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict()
    }
    
    # Numeric analysis
    stats_df, corr_matrix, skewness = analyze_numeric_features(df)
    report['basic_stats'] = stats_df
    report['correlation_matrix'] = corr_matrix
    report['skewness'] = skewness
    
    # Categorical analysis
    report['categorical_stats'] = analyze_categorical_features(df)
    
    # Key features analysis
    if target_cols:
        report['key_features'] = identify_key_features(corr_matrix, target_cols)
    
    return report

def print_feature_report(report: Dict[str, Any]) -> None:
    """
    Print a formatted feature analysis report.
    
    Args:
        report (Dict[str, Any]): Feature analysis report dictionary
    """
    print(f"\n{'='*20} {report['game_name']} Analysis {'='*20}")
    print(f"\nDataset Shape: {report['shape']}")
    
    print("\nMissing Values:")
    for col, count in report['missing_values'].items():
        if count > 0:
            print(f"{col}: {count}")
    
    print("\nBasic Statistics:")
    print(report['basic_stats'])
    
    print("\nFeature Skewness:")
    print(report['skewness'])
    
    if 'key_features' in report:
        print("\nKey Feature Correlations:")
        for target, correlations in report['key_features'].items():
            print(f"\nStrong correlations with {target}:")
            print(correlations)
    
    print("\nCategorical Feature Distributions:")
    for col, counts in report['categorical_stats'].items():
        print(f"\n{col}:")
        for value, count in counts.items():
            print(f"  {value}: {count}")

def normalize_features(df: pd.DataFrame, 
                      exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize numeric features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        exclude_cols (List[str], optional): Columns to exclude from normalization
        
    Returns:
        Tuple containing:
        - Normalized DataFrame
        - Fitted StandardScaler object
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Select numeric columns for normalization
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Create copy of DataFrame
    normalized_df = df.copy()
    
    # Normalize selected columns
    normalized_df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    return normalized_df, scaler