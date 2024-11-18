import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def remove_zero_stat_players(df, stat_columns):
    """
    Remove rows from a DataFrame where all specified stat columns have a value of 0.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing player stats
    stat_columns (list): List of column names to check for zeros. 
                        If None, uses all numeric columns except index
    
    Returns:
    pandas.DataFrame: DataFrame with zero-stat players removed
    
    Source: Shayne's players_n_stats.ipynb
    """
    # If no stat columns specified, use all numeric columns
    if stat_columns is None:
        stat_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create a boolean mask where True means the row has all zeros in stat columns
    zero_mask = df[stat_columns].eq(0).all(axis=1)
    
    # Return DataFrame with zero-stat players removed
    return df[~zero_mask]

def parse_nested_dict(d: str) -> Dict[str, Any]:
    """Convert string representation of dictionary to actual dictionary."""
    if isinstance(d, str):
        try:
            return eval(d)
        except:
            return {}
    return d if d is not None else {}

def extract_segment_stats(segments: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Extract statistics from segments list."""
    stats = {}
    raw_data_mask = {}

    if not segments:
        return {}, {}

    # Accumulate stats across all segments
    total_rounds = 0
    total_kills = 0
    total_deaths = 0
    total_win_rate = 0
    num_segments_with_win_rate = 0

    # Objectives dictionary to accumulate objective counts across all segments
    objectives_stats = {
        'defuse_with_kit_total': 0,
        'defuse_without_kit_total': 0,
        'defuse_bomb_total': 0,
        'explode_bomb_total': 0,
        'plant_bomb_total': 0
    }

    for segment in segments:
        if segment['type'] == 'round':
            # Extract round information
            total_rounds += segment.get('count', 0)

            # Extract combat stats
            combat = segment.get('combat', {})
            total_kills += combat.get('kills', {}).get('total', 0)
            total_deaths += combat.get('deaths', {}).get('total', 0)

            # Extract win rate if available
            if 'win_rate' in segment:
                total_win_rate += segment['win_rate']
                num_segments_with_win_rate += 1

            # Extract objectives
            objectives = segment.get('objectives', [])
            for obj in objectives:
                obj_type = obj.get('type')
                if obj_type == 'beginDefuseWithKit':
                    objectives_stats['defuse_with_kit_total'] += obj.get('completions', 0)
                elif obj_type == 'beginDefuseWithoutKit':
                    objectives_stats['defuse_without_kit_total'] += obj.get('completions', 0)
                elif obj_type == 'defuseBomb':
                    objectives_stats['defuse_bomb_total'] += obj.get('completions', 0)
                elif obj_type == 'explodeBomb':
                    objectives_stats['explode_bomb_total'] += obj.get('completions', 0)
                elif obj_type == 'plantBomb':
                    objectives_stats['plant_bomb_total'] += obj.get('completions', 0)

    # Add combat stats to the output dictionary
    stats.update({
        'total_rounds': total_rounds,
        'total_segment_kills': total_kills,
        'total_segment_deaths': total_deaths
    })

    # Add per-round averages if applicable
    if total_rounds > 0:
        stats.update({
            'avg_kills_per_round': total_kills / total_rounds,
            'avg_deaths_per_round': total_deaths / total_rounds
        })

    # Add win rate if any segments have win rate data
    if num_segments_with_win_rate > 0:
        stats['avg_win_rate'] = total_win_rate / num_segments_with_win_rate
    else:
        stats['avg_win_rate'] = 0

    # Add objectives to the output dictionary
    stats.update(objectives_stats)

    return stats, raw_data_mask

def extract_nested_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts basic nested statistics from the dataframe.
    """
    flat_data = []
    for _, row in df.iterrows():
        flat_row = {'player_id': row['player_id']}
        
        # Extract General Stats
        general = parse_nested_dict(row['general'])
        flat_row.update({
            'series_played': general.get('series_played', 0),
            'games_played': general.get('games_played', 0)
        })
        
        # Extract Combat Stats
        combat = parse_nested_dict(row['combat'])
        kills = combat.get('kills', {})
        deaths = combat.get('deaths', {})
        flat_row.update({
            'total_kills': kills.get('total', 0),
            'avg_kills': kills.get('average', 0),
            'best_kills': kills.get('best', 0),
            'total_deaths': deaths.get('total', 0),
            'avg_deaths': deaths.get('average', 0),
            'first_kills_total': combat.get('first_kills', {}).get('total', 0),
            'first_kills_percentage': combat.get('first_kills', {}).get('percentage', 0),
            'damage_total': combat.get('damage', {}).get('total', 0),
            'damage_avg': combat.get('damage', {}).get('average', 0),
            'damage_max': combat.get('damage', {}).get('max', 0)
        })
        
        # Extract Performance Stats
        performance = parse_nested_dict(row['performance'])
        wins = performance.get('wins', {})
        flat_row.update({
            'wins_count': wins.get('count', 0),
            'win_percentage': wins.get('percentage', 0),
            'current_streak': wins.get('current_streak', 0)
        })
        
        # Extract Economy Stats
        economy = parse_nested_dict(row['economy'])
        net_worth = economy.get('net_worth', {})
        flat_row.update({
            'avg_net_worth': net_worth.get('average', 0),
            'max_net_worth': net_worth.get('max', 0)
        })

        # Handle Progression and Segments
        progression = parse_nested_dict(row['progression'])
        if isinstance(progression, dict) and len(progression) > 0:
            # If progression is present, extract progression stats
            flat_row.update({
                'experience_total': progression.get('experience', {}).get('total', 0),
                'experience_avg': progression.get('experience', {}).get('average', 0),
                'experience_max': progression.get('experience', {}).get('max', 0)
            })
        else:
            # If segments are present, extract segment stats
            segments = parse_nested_dict(row['segments'])
            segment_stats, _ = extract_segment_stats(segments)
            flat_row.update(segment_stats)
        
        # Add flattened row to list
        flat_data.append(flat_row)
    
    return pd.DataFrame(flat_data)

def split_dataframe_by_title_id(df: pd.DataFrame) -> dict:
    """
    Splits the input DataFrame into separate DataFrames based on the title_id.
    Each resulting DataFrame corresponds to a unique title_id found in the original DataFrame.

    Args:
        df (pd.DataFrame): The merged DataFrame containing player data across multiple games.

    Returns:
        dict: A dictionary where keys are title_ids and values are DataFrames corresponding to each title_id.
    """
    # Get a list of unique title IDs from the merged DataFrame
    unique_title_ids = df['title_id'].unique()
    
    # Create a dictionary to store DataFrames for each title ID
    title_dataframes = {}
    
    # Iterate through each unique title ID and create a separate DataFrame for each
    for title_id in unique_title_ids:
        title_dataframes[title_id] = df[df['title_id'] == title_id].reset_index(drop=True)
    
    return title_dataframes