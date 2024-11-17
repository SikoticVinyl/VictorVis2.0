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

def extract_nested_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy function maintained for backwards compatibility.
    Extracts basic nested statistics from the dataframe.
    """
    flat_data = []
    for _, row in df.iterrows():
        flat_row = {'player_id': row['player_id']}
        
        general = parse_nested_dict(row['general'])
        flat_row.update({
            'series_played': general.get('series_played', 0),
            'games_played': general.get('games_played', 0)
        })
        
        combat = parse_nested_dict(row['combat'])
        kills = combat.get('kills', {})
        deaths = combat.get('deaths', {})
        flat_row.update({
            'total_kills': kills.get('total', 0),
            'avg_kills': kills.get('average', 0),
            'best_kills': kills.get('best', 0),
            'total_deaths': deaths.get('total', 0),
            'avg_deaths': deaths.get('average', 0)
        })
        
        performance = parse_nested_dict(row['performance'])
        wins = performance.get('wins', {})
        flat_row.update({
            'wins_count': wins.get('count', 0),
            'win_percentage': wins.get('percentage', 0),
            'current_streak': wins.get('current_streak', 0)
        })
        
        economy = parse_nested_dict(row['economy'])
        net_worth = economy.get('net_worth', {})
        flat_row.update({
            'avg_net_worth': net_worth.get('average', 0),
            'max_net_worth': net_worth.get('max', 0)
        })
        
        flat_data.append(flat_row)
    
    return pd.DataFrame(flat_data)

def extract_combat_stats(combat_dict: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Extract combat statistics from nested dictionary."""
    stats = {}
    raw_data_mask = {}
    
    # Extract kills information
    if 'kills' in combat_dict:
        kills = combat_dict['kills']
        for key, stat_name in [
            ('total', 'total_kills'),
            ('average', 'avg_kills'),
            ('best', 'best_kills'),
            ('worst', 'worst_kills')
        ]:
            stats[stat_name] = kills.get(key, np.nan)
            raw_data_mask[stat_name] = key in kills
        
    # Extract deaths information
    if 'deaths' in combat_dict:
        deaths = combat_dict['deaths']
        for key, stat_name in [
            ('total', 'total_deaths'),
            ('average', 'avg_deaths'),
            ('best', 'best_deaths'),
            ('worst', 'worst_deaths')
        ]:
            stats[stat_name] = deaths.get(key, np.nan)
            raw_data_mask[stat_name] = key in deaths
    
    # Extract first kills
    if 'first_kills' in combat_dict:
        first_kills = combat_dict['first_kills']
        stats.update({
            'total_first_kills': first_kills.get('total', np.nan),
            'first_kill_percentage': first_kills.get('percentage', np.nan)
        })
        raw_data_mask.update({
            'total_first_kills': 'total' in first_kills,
            'first_kill_percentage': 'percentage' in first_kills
        })
    
    # Extract damage if available (R6 specific)
    if 'damage' in combat_dict:
        damage = combat_dict['damage']
        stats.update({
            'total_damage': damage.get('total', np.nan),
            'avg_damage': damage.get('average', np.nan),
            'max_damage': damage.get('max', np.nan)
        })
        raw_data_mask.update({
            'total_damage': 'total' in damage,
            'avg_damage': 'average' in damage,
            'max_damage': 'max' in damage
        })
    
    return stats, raw_data_mask

def extract_segment_stats(segments: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Extract statistics from segments list."""
    stats = {}
    raw_data_mask = {}
    
    if not segments:
        return {}, {}
    
    for segment in segments:
        if segment['type'] == 'round':
            # Basic round stats
            stats['round_count'] = segment['count']
            raw_data_mask['round_count'] = True
            
            # Win rate from segments
            stats['round_win_rate'] = segment.get('win_rate', np.nan)
            raw_data_mask['round_win_rate'] = 'win_rate' in segment
            
            # Combat stats from segments
            combat = segment.get('combat', {})
            kills = combat.get('kills', {})
            deaths = combat.get('deaths', {})
            
            stats.update({
                'round_kills_total': kills.get('total', np.nan),
                'round_kills_avg': kills.get('average', np.nan),
                'round_deaths_total': deaths.get('total', np.nan),
                'round_deaths_avg': deaths.get('average', np.nan)
            })
            
            raw_data_mask.update({
                'round_kills_total': 'total' in kills,
                'round_kills_avg': 'average' in kills,
                'round_deaths_total': 'total' in deaths,
                'round_deaths_avg': 'average' in deaths
            })
            
            # Process objectives
            objectives = segment.get('objectives', [])
            for obj in objectives:
                obj_type = obj['type']
                base_name = {
                    'beginDefuseWithKit': 'defuse_with_kit',
                    'beginDefuseWithoutKit': 'defuse_without_kit',
                    'explodeBomb': 'bomb_explosions',
                    'plantBomb': 'bomb_plants'
                }.get(obj_type)
                
                if base_name:
                    stats[f'{base_name}_total'] = obj.get('completions', np.nan)
                    raw_data_mask[f'{base_name}_total'] = 'completions' in obj
                    
                    first_completions = obj.get('first_completions', {})
                    if first_completions:
                        stats[f'{base_name}_first_percentage'] = first_completions.get('percentage', np.nan)
                        raw_data_mask[f'{base_name}_first_percentage'] = 'percentage' in first_completions
    
    # Calculate per-round metrics
    if 'round_count' in stats and stats['round_count'] > 0:
        for base_stat in ['kills', 'deaths']:
            if f'round_{base_stat}_total' in stats:
                per_round = f'round_{base_stat}_per_round'
                stats[per_round] = stats[f'round_{base_stat}_total'] / stats['round_count']
                raw_data_mask[per_round] = raw_data_mask[f'round_{base_stat}_total']
        
        # Objective per round calculations
        for obj_type in ['defuse_with_kit', 'defuse_without_kit', 'bomb_explosions', 'bomb_plants']:
            if f'{obj_type}_total' in stats:
                per_round = f'{obj_type}_per_round'
                stats[per_round] = stats[f'{obj_type}_total'] / stats['round_count']
                raw_data_mask[per_round] = raw_data_mask[f'{obj_type}_total']
    
    return stats, raw_data_mask

def process_player_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process player statistics and return processed data with data availability mask."""
    processed_data = []
    raw_data_masks = []
    
    for _, row in df.iterrows():
        player_data = {'player_id': row['player_id']}
        data_mask = {'player_id': True}
        
        # Extract general stats
        general = parse_nested_dict(row['general'])
        player_data.update({
            'series_played': general.get('series_played', np.nan),
            'games_played': general.get('games_played', np.nan)
        })
        data_mask.update({
            'series_played': 'series_played' in general,
            'games_played': 'games_played' in general
        })
        
        # Extract combat stats
        combat_stats, combat_mask = extract_combat_stats(parse_nested_dict(row['combat']))
        player_data.update(combat_stats)
        data_mask.update(combat_mask)
        
        # Extract segment stats
        segment_stats, segment_mask = extract_segment_stats(parse_nested_dict(row['segments']))
        player_data.update(segment_stats)
        data_mask.update(segment_mask)
        
        # Extract progression if available
        progression = parse_nested_dict(row['progression'])
        if 'experience' in progression:
            exp = progression['experience']
            player_data.update({
                'total_experience': exp.get('total', np.nan),
                'avg_experience': exp.get('average', np.nan),
                'max_experience': exp.get('max', np.nan)
            })
            data_mask.update({
                'total_experience': 'total' in exp,
                'avg_experience': 'average' in exp,
                'max_experience': 'max' in exp
            })
        
        processed_data.append(player_data)
        raw_data_masks.append(data_mask)
    
    return pd.DataFrame(processed_data), pd.DataFrame(raw_data_masks)