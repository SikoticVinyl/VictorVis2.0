import os
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

class EnhancedStatsCollector:
    def __init__(self, api_key: str):
        """
        Initialize the Enhanced Stats Collector.
        
        Args:
            api_key (str): The API key for Grid.gg
        """
        self.api_key = api_key
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create handlers if they don't exist
        if not self.logger.handlers:
            c_handler = logging.StreamHandler()
            f_handler = logging.FileHandler('collector.log')
            c_handler.setLevel(logging.INFO)
            f_handler.setLevel(logging.INFO)
            
            # Create formatters and add to handlers
            log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(log_format)
            f_handler.setFormatter(log_format)
            
            # Add handlers to the logger
            self.logger.addHandler(c_handler)
            self.logger.addHandler(f_handler)
        
        # Initialize clients
        self.stats_client = self._create_client('https://api-op.grid.gg/statistics-feed/graphql')
        self.central_client = self._create_client('https://api-op.grid.gg/central-data/graphql')
        
        # Set up paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.queries_dir = os.path.join(self.base_dir, 'queries')
        
        # Ensure queries directory exists
        if not os.path.exists(self.queries_dir):
            os.makedirs(self.queries_dir)
            self.logger.info(f"Created queries directory at {self.queries_dir}")
            
        # Create data directory for outputs if it doesn't exist
        self.data_dir = os.path.join(self.base_dir, 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.logger.info(f"Created data directory at {self.data_dir}")

    def _create_client(self, url: str) -> Client:
        """Create a GraphQL client with proper headers and retry logic."""
        transport = RequestsHTTPTransport(
            url=url,
            headers={'x-api-key': self.api_key},
            retries=3
        )
        return Client(transport=transport, fetch_schema_from_transport=True)

    def _load_query(self, filename: str) -> str:
        """Load a GraphQL query from file."""
        query_path = os.path.join(self.queries_dir, f'{filename}.graphql')
        try:
            with open(query_path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Query file not found at: {query_path}")

    def _save_to_file(self, df: pd.DataFrame, prefix: str) -> None:
        """Save DataFrame to CSV with timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'{prefix}_{timestamp}.csv'
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved data to {filepath}")

    def fetch_all_players(self) -> pd.DataFrame:
        """
        Fetch all players across all titles with enhanced information.
        """
        query = self._load_query('enhanced_players')
        
        all_players = []
        has_next_page = True
        cursor = None
        page_count = 0
        
        while has_next_page:
            try:
                variables = {
                    "after": cursor
                }
                
                self.logger.info(f"Fetching page {page_count + 1} of players")
                result = self.central_client.execute(gql(query), variable_values=variables)
                
                # Extract player data
                for edge in result['players']['edges']:
                    player = edge['node']
                    processed_player = {
                        'id': player['id'],
                        'nickname': player['nickname'],
                        'full_name': f"{player.get('firstName', '')} {player.get('lastName', '')}".strip(),
                        'nationality': player.get('nationality'),
                        'team_id': player.get('team', {}).get('id'),
                        'team_name': player.get('team', {}).get('name'),
                        'team_short_name': player.get('team', {}).get('shortName'),
                        'title_id': player.get('title', {}).get('id'),
                        'title_name': player.get('title', {}).get('name'),
                        'title_short_name': player.get('title', {}).get('nameShortened'),
                        'active': player.get('active', False),
                        'primary_role': next((role['name'] for role in player.get('roles', []) 
                                           if role.get('primary')), None),
                        'games_played': player.get('statistics', {}).get('gamesPlayed', 0),
                        'last_game_date': player.get('statistics', {}).get('lastGameDate')
                    }
                    all_players.append(processed_player)
                
                # Update pagination
                page_info = result['players']['pageInfo']
                has_next_page = page_info['hasNextPage']
                cursor = page_info['endCursor']
                
                # Log progress
                page_count += 1
                self.logger.info(f"Processed page {page_count} - Total players so far: {len(all_players)}")
                
                # Save intermediate results every 10 pages
                if page_count % 10 == 0 and all_players:
                    temp_df = pd.DataFrame(all_players)
                    self._save_to_file(temp_df, f'players_temp_page{page_count}')
                
                # Add delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error fetching players on page {page_count}: {str(e)}")
                # Save what we have so far in case of error
                if all_players:
                    error_df = pd.DataFrame(all_players)
                    self._save_to_file(error_df, 'players_error_recovery')
                break
        
        # Create final DataFrame
        df = pd.DataFrame(all_players)
        
        # Convert dates if present
        if 'last_game_date' in df.columns:
            df['last_game_date'] = pd.to_datetime(df['last_game_date'], errors='coerce')
        
        # Save final results
        if not df.empty:
            self._save_to_file(df, 'players_final')
        
        # Log collection summary
        self.logger.info(f"""
        Player Collection Summary:
        -------------------------
        Total Players: {len(df)}
        Active Players: {len(df[df['active'] == True])}
        Players with Teams: {len(df[df['team_id'].notna()])}
        Unique Titles: {df['title_name'].nunique()}
        Unique Teams: {df['team_name'].nunique()}
        """)
        
        return df

    def get_player_comprehensive_stats(self, player_id: str) -> Dict[str, Any]:
        """
        Get detailed player statistics with enhanced error handling and game-specific stats.
        """
        query = self._load_query('comprehensive_player_stats')
        
        try:
            variables = {
                'playerId': player_id,
                'filter': {
                    'timeWindow': 'LAST_3_MONTHS'
                }
            }
            
            self.logger.info(f"Fetching comprehensive stats for player {player_id}")
            result = self.stats_client.execute(gql(query), variable_values=variables)
            
            if not result or 'playerStatistics' not in result:
                self.logger.warning(f"No statistics found for player {player_id}")
                return {}
            
            # Log raw data for debugging
            self.logger.info(f"Raw stats structure for player {player_id}:")
            self.logger.info(f"Result structure: {result}")
            
            stats = result['playerStatistics']
            return self._process_player_stats(stats, player_id)
            
        except Exception as e:
            self.logger.error(f"Error fetching stats for player {player_id}: {str(e)}")
            return {}

    def _process_player_stats(self, stats: Dict[str, Any], player_id: str) -> Dict[str, Any]:
        """Process raw player statistics into a structured format."""
        game_stats = stats.get('game', {})
        
        # Log raw stats for debugging
        self.logger.info(f"Processing stats for player {player_id}")
        self.logger.info(f"Series stats: {stats.get('series', {})}")
        self.logger.info(f"Game stats: {game_stats}")
        
        processed_stats = {
            'player_id': player_id,
            'general': {
                'series_played': stats.get('series', {}).get('count'),
                'games_played': game_stats.get('count')
            },
            'combat': {
                'kills': {
                    'total': stats.get('series', {}).get('kills', {}).get('sum'),
                    'average': stats.get('series', {}).get('kills', {}).get('avg'),
                    'best': stats.get('series', {}).get('kills', {}).get('max'),
                    'worst': stats.get('series', {}).get('kills', {}).get('min')
                },
                'deaths': {
                    'total': stats.get('series', {}).get('deaths', {}).get('sum'),
                    'average': stats.get('series', {}).get('deaths', {}).get('avg'),
                    'worst': stats.get('series', {}).get('deaths', {}).get('max'),
                    'best': stats.get('series', {}).get('deaths', {}).get('min')
                }
            }
        }
        
        # Log processed base stats
        self.logger.info(f"Processed base stats: {processed_stats}")
        
        # Add first kill statistics if available
        if 'firstKill' in stats.get('series', {}):
            first_kill_stats = stats['series']['firstKill']
            self.logger.info(f"First kill stats: {first_kill_stats}")
            if first_kill_stats and isinstance(first_kill_stats, list):
                processed_stats['combat']['first_kills'] = {
                    'count': first_kill_stats[0].get('count'),
                    'percentage': first_kill_stats[0].get('percentage')
                }
        
        # Add game-specific stats
        if isinstance(game_stats, dict):
            # Add win statistics
            if 'won' in game_stats and isinstance(game_stats['won'], list) and game_stats['won']:
                win_stats = game_stats['won'][0]
                self.logger.info(f"Win stats: {win_stats}")
                processed_stats['performance'] = {
                    'wins': {
                        'count': win_stats.get('count'),
                        'percentage': win_stats.get('percentage'),
                        'current_streak': win_stats.get('streak', {}).get('current'),
                        'max_streak': win_stats.get('streak', {}).get('max')
                    }
                }
            
            # Add economic statistics
            econ_stats = {}
            if 'money' in game_stats:
                self.logger.info(f"Money stats: {game_stats['money']}")
                econ_stats['money'] = {
                    'total': game_stats['money'].get('sum'),
                    'average': game_stats['money'].get('avg'),
                    'max': game_stats['money'].get('max')
                }
            if 'inventoryValue' in game_stats:
                self.logger.info(f"Inventory stats: {game_stats['inventoryValue']}")
                econ_stats['inventory_value'] = {
                    'average': game_stats['inventoryValue'].get('avg'),
                    'max': game_stats['inventoryValue'].get('max')
                }
            if 'netWorth' in game_stats:
                self.logger.info(f"Net worth stats: {game_stats['netWorth']}")
                econ_stats['net_worth'] = {
                    'average': game_stats['netWorth'].get('avg'),
                    'max': game_stats['netWorth'].get('max')
                }
            if econ_stats:
                processed_stats['economy'] = econ_stats
            
            # Add damage stats (CS2, R6)
            if 'damageDealt' in game_stats:
                self.logger.info(f"Damage stats: {game_stats['damageDealt']}")
                processed_stats['combat']['damage'] = {
                    'total': game_stats['damageDealt'].get('sum'),
                    'average': game_stats['damageDealt'].get('avg'),
                    'max': game_stats['damageDealt'].get('max')
                }
            
            # Add MOBA-specific stats
            if 'experiencePoints' in game_stats:
                self.logger.info(f"Experience stats: {game_stats['experiencePoints']}")
                processed_stats['progression'] = {
                    'experience': {
                        'total': game_stats['experiencePoints'].get('sum'),
                        'average': game_stats['experiencePoints'].get('avg'),
                        'max': game_stats['experiencePoints'].get('max')
                    }
                }
            
            # Add LOL-specific stats
            if 'totalMoneyEarned' in game_stats:
                self.logger.info(f"Total money earned stats: {game_stats['totalMoneyEarned']}")
                if 'economy' not in processed_stats:
                    processed_stats['economy'] = {}
                processed_stats['economy']['total_earned'] = {
                    'total': game_stats['totalMoneyEarned'].get('sum'),
                    'average': game_stats['totalMoneyEarned'].get('avg'),
                    'max': game_stats['totalMoneyEarned'].get('max')
                }

            # Add R6-specific stats
            if 'healingDealt' in game_stats or 'healingReceived' in game_stats:
                support_stats = {}
                if 'healingDealt' in game_stats:
                    self.logger.info(f"Healing dealt stats: {game_stats['healingDealt']}")
                    support_stats['healing_dealt'] = {
                        'total': game_stats['healingDealt'].get('sum'),
                        'average': game_stats['healingDealt'].get('avg')
                    }
                if 'healingReceived' in game_stats:
                    self.logger.info(f"Healing received stats: {game_stats['healingReceived']}")
                    support_stats['healing_received'] = {
                        'total': game_stats['healingReceived'].get('sum'),
                        'average': game_stats['healingReceived'].get('avg')
                    }
                if support_stats:
                    processed_stats['support'] = support_stats
        
            # Add unit kills if available
            if 'unitKills' in game_stats:
                self.logger.info(f"Unit kills stats: {game_stats['unitKills']}")
                if isinstance(game_stats['unitKills'], list):
                    unit_kills = {}
                    for unit in game_stats['unitKills']:
                        unit_name = unit.get('unitName')
                        if unit_name:
                            unit_kills[unit_name] = {
                                'count': unit.get('count', {}).get('sum'),
                                'average': unit.get('count', {}).get('avg')
                            }
                    if unit_kills:
                        processed_stats['combat']['unit_kills'] = unit_kills

        # Process segment statistics if available
        if 'segment' in stats and isinstance(stats['segment'], list):
            self.logger.info(f"Processing segment stats")
            processed_stats['segments'] = []
            for segment in stats['segment']:
                self.logger.info(f"Segment data: {segment}")
                segment_data = {
                    'type': segment.get('type'),
                    'count': segment.get('count'),
                    'combat': {
                        'kills': {
                            'total': segment.get('kills', {}).get('sum'),
                            'average': segment.get('kills', {}).get('avg')
                        },
                        'deaths': {
                            'total': segment.get('deaths', {}).get('sum'),
                            'average': segment.get('deaths', {}).get('avg')
                        }
                    }
                }
                
                # Add win rate if available
                if segment.get('won') and isinstance(segment['won'], list):
                    segment_data['win_rate'] = segment['won'][0].get('percentage')
                
                # Add objectives if available
                if 'objectives' in segment and isinstance(segment['objectives'], list):
                    segment_data['objectives'] = []
                    for objective in segment['objectives']:
                        obj_data = {
                            'type': objective.get('type'),
                            'completions': objective.get('completionCount', {}).get('sum'),
                            'avg_completions': objective.get('completionCount', {}).get('avg')
                        }
                        
                        # Add first completion stats if available
                        if 'completedFirst' in objective and isinstance(objective['completedFirst'], list):
                            obj_data['first_completions'] = {
                                'count': objective['completedFirst'][0].get('count'),
                                'percentage': objective['completedFirst'][0].get('percentage')
                            }
                        
                        segment_data['objectives'].append(obj_data)
                
                processed_stats['segments'].append(segment_data)
                
        # Final log of complete processed stats
        self.logger.info(f"Final processed stats structure for player {player_id}:")
        self.logger.info(f"Complete stats: {processed_stats}")
        
        return processed_stats

    def collect_stats_for_all_players(self, min_games: int = 10, batch_size: int = 50) -> pd.DataFrame:
        """
        Collect comprehensive statistics for all players meeting minimum game threshold.
        
        Args:
            min_games (int): Minimum number of games played to include player
            batch_size (int): Number of players to process in each batch
            
        Returns:
            pd.DataFrame: DataFrame containing all player statistics
        """
        # First, get all players
        players_df = self.fetch_all_players()
        
        # Filter for players with minimum games
        active_players = players_df[players_df['games_played'] >= min_games].copy()
        
        self.logger.info(f"""
        Player Stats Collection Plan:
        ---------------------------
        Total players found: {len(players_df)}
        Players with {min_games}+ games: {len(active_players)}
        Number of batches: {len(active_players) // batch_size + 1}
        """)
        
        all_stats = []
        total_players = len(active_players)
        
        # Process in batches
        for i in range(0, total_players, batch_size):
            batch = active_players.iloc[i:i + batch_size]
            batch_num = i//batch_size + 1
            total_batches = (total_players + batch_size - 1)//batch_size
            
            self.logger.info(f"Processing batch {batch_num} of {total_batches}")
            
            batch_stats = []
            for _, player in batch.iterrows():
                try:
                    stats = self.get_player_comprehensive_stats(player['id'])
                    if stats:
                        # Combine player info with stats
                        stats['player_info'] = player.to_dict()
                        batch_stats.append(stats)
                        self.logger.debug(f"Successfully processed player {player['nickname']}")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing player {player['nickname']}: {str(e)}")
                    continue
                
                # Add small delay between requests
                time.sleep(0.5)
            
            # Add batch stats to overall stats
            all_stats.extend(batch_stats)
            
            # Save intermediate results
            if batch_stats:
                batch_df = pd.DataFrame(batch_stats)
                self._save_to_file(batch_df, f'player_stats_batch_{batch_num}_of_{total_batches}')
                
                self.logger.info(f"""
                Batch {batch_num} Summary:
                -------------------------
                Players processed: {len(batch_stats)}
                Success rate: {len(batch_stats)/len(batch)*100:.1f}%
                Total players processed so far: {len(all_stats)}
                """)
        
        # Create final DataFrame
        final_df = pd.DataFrame(all_stats) if all_stats else pd.DataFrame()
        
        if not final_df.empty:
            # Save final results
            self._save_to_file(final_df, 'player_stats_final')
            
            self.logger.info(f"""
            Final Collection Summary:
            -----------------------
            Total players processed: {len(final_df)}
            Total success rate: {len(final_df)/len(active_players)*100:.1f}%
            """)
        
        return final_df