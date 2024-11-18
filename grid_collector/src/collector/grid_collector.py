import os
import time
import logging
from datetime import datetime, timedelta
from typing import Union, Dict, List, Optional, Any

import pandas as pd
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

from ..utils import RateLimiter, GridAPIError, handle_api_error
from .base import BaseCollector

class GridCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = RateLimiter()
        self.logger = logging.getLogger(__name__)
        
        # Add this to get the absolute path to your project root
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Initialize clients
        self.central_client = self._create_client('https://api-op.grid.gg/central-data/graphql')
        self.stats_client = self._create_client('https://api-op.grid.gg/statistics-feed/graphql')

    # Private utility methods
    def _create_client(self, url: str) -> Client:
        """Create a GraphQL client for a specific endpoint."""
        transport = RequestsHTTPTransport(
            url=url,
            headers={'x-api-key': self.api_key},
            retries=3
        )
        return Client(transport=transport, fetch_schema_from_transport=True)

    def _load_query(self, filename: str) -> str:
        """Load a GraphQL query from file."""
        query_path = os.path.join(self.base_dir, 'queries', f'{filename}.graphql')
        self.logger.info(f"Looking for query file at: {query_path}")
        
        try:
            with open(query_path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Query file not found at: {query_path}")

    def _execute_query(self, query: str, variables: Dict[str, Any], client: Client) -> Dict[str, Any]:
        """Execute a single query with rate limiting and error handling."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Wait for rate limit - This uses the RateLimiter from utils
                self.rate_limiter.wait()
                
                # Execute query
                self.logger.debug(f"Executing query (attempt {attempt + 1})")
                result = client.execute(gql(query), variable_values=variables)
                
                return result
                
            except Exception as e:
                if 'ENHANCE_YOUR_CALM' in str(e) or 'rate limit' in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        self.logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry")
                        time.sleep(wait_time)
                        continue
                # Use the error handler from utils
                raise handle_api_error(e)

    def _execute_paginated_query(self, query: str, client: Client, variables: Dict = None, limit: int = None) -> List[Dict]:
        """Execute a paginated query and return all nodes, stopping when limit is reached."""
        if variables is None:
            variables = {}
        
        all_nodes = []
        has_next_page = True
        after = None
        page_number = 1
    
        while has_next_page:
            try:
                # Update variables for pagination
                current_vars = {**variables, 'first': 50, 'after': after}
            
                # Execute query with rate limiting
                result = self._execute_query(query, current_vars, client)
            
                # Get the correct data key (first key in result)
                data_key = next(key for key in result.keys() if key != '__typename')
                data = result[data_key]
            
                # Extract edges and nodes
                edges = data.get('edges', [])
                for edge in edges:
                    if 'node' in edge:
                        all_nodes.append(edge['node'])
                    
                        # Check if limit is reached
                        if limit is not None and len(all_nodes) >= limit:
                            self.logger.info(f"Limit of {limit} items reached.")
                            return all_nodes[:limit]
            
               # Update pagination info
                page_info = data.get('pageInfo', {})
                has_next_page = page_info.get('hasNextPage', False)
                after = page_info.get('endCursor') if has_next_page else None
            
                # Log progress
                self.logger.info(f"Processed page {page_number} - Got {len(edges)} items")
                page_number += 1
            
                # Add a small delay between pages to help prevent rate limiting
                if has_next_page:
                    time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error during pagination on page {page_number}: {str(e)}")
                raise
                
        return all_nodes


    def get_tournaments(self) -> pd.DataFrame:
        """Get all tournaments."""
        query = self._load_query('tournaments')
        
        try:
            # Get all tournament nodes
            tournaments = self._execute_paginated_query(query, self.central_client)
            
            # Process tournaments into a list of dictionaries
            processed_tournaments = []
            for tournament in tournaments:
                # Extract title information
                titles = tournament.get('titles', [])
                title_ids = [t.get('id') for t in titles]
                title_names = [t.get('name') for t in titles]
                
                # Create tournament record
                tournament_data = {
                    'id': tournament.get('id'),
                    'name': tournament.get('name'),
                    'name_short': tournament.get('nameShortened'),
                    'start_date': tournament.get('startDate'),
                    'end_date': tournament.get('endDate'),
                    'private': tournament.get('private', False),
                    'title_ids': title_ids,
                    'title_names': title_names
                }
                processed_tournaments.append(tournament_data)
            
            # Convert to DataFrame
            df = pd.DataFrame(processed_tournaments)
            
            # Convert dates
            for date_col in ['start_date', 'end_date']:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting tournament data: {str(e)}")
            raise

    def get_matches(self, days: int = 7) -> pd.DataFrame:
        """Get recent matches based on the series data from the API

        Args:
            days (int): Number of days of match history to collect
        
        Returns:
            pd.DataFrame: DataFrame containing match information
        """
        query = self._load_query('matches')
    
        # Calculate date range and format in the exact format the API expects
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
    
        # Format dates exactly like the API example
        formatted_start = start_date.strftime('%Y-%m-%dT%H:00:00+00:00')
        formatted_end = end_date.strftime('%Y-%m-%dT%H:00:00+00:00')
    
        # Replace the date placeholders in the query
        modified_query = query.replace(
            '"2024-10-30T14:00:00+00:00"',
            f'"{formatted_start}"'
        ).replace(
            '"2024-11-06T14:00:00+00:00"',
            f'"{formatted_end}"'
        )
    
        variables = {
            'first': 50,
            'after': None
        }
    
        try:
            matches = self._execute_paginated_query(modified_query, self.central_client, variables)
            processed_matches = []
        
            for match in matches:
                if len(match.get('teams', [])) >= 2:
                    match_data = {
                        'id': match['id'],
                        'series_title': match.get('title', {}).get('nameShortened'),
                        'tournament_name': match.get('tournament', {}).get('nameShortened'),
                        'start_time': match['startTimeScheduled'],
                        'format_name': match.get('format', {}).get('name'),
                        'format_short': match.get('format', {}).get('nameShortened'),
                    
                        # Team 1 information
                        'team1_id': match['teams'][0]['baseInfo'].get('id'),
                        'team1_name': match['teams'][0]['baseInfo'].get('name'),
                        'team1_score_advantage': match['teams'][0].get('scoreAdvantage'),
                    
                        # Team 2 information
                        'team2_id': match['teams'][1]['baseInfo'].get('id'),
                        'team2_name': match['teams'][1]['baseInfo'].get('name'),
                        'team2_score_advantage': match['teams'][1].get('scoreAdvantage')
                    }
                    processed_matches.append(match_data)
        
            # Convert to DataFrame
            df = pd.DataFrame(processed_matches)
        
            # Convert datetime columns
            if 'start_time' in df.columns:
                df['start_time'] = pd.to_datetime(df['start_time'])
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error collecting match data: {str(e)}")
            raise
    
    def get_all_titles(self) -> pd.DataFrame:
        """Fetch all available titles and return as a DataFrame."""
        query = self._load_query('titles')
        variables = {}  # No filter, fetch all titles
        try:
            result = self._execute_query(query, variables, self.central_client)
            titles = result.get('titles', [])
            # Process titles into a list of dictionaries
            processed_titles = []
            for title in titles:
                title_data = {
                    'id': title.get('id'),
                    'name': title.get('name'),
                    'name_shortened': title.get('nameShortened'),
                    'private': title.get('private')
                }
                processed_titles.append(title_data)
            # Convert to DataFrame
            df = pd.DataFrame(processed_titles)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching titles: {str(e)}")
            return pd.DataFrame()

    def get_players(self, limit: int = None, title_id: str = None) -> pd.DataFrame:
        """
        Get players with specified fields, optionally limiting the number of players fetched.
        
        Args:
            limit (int, optional): Maximum number of players to fetch
            title_id (str, optional): Filter by specific title ID. If None, gets players from all titles.
            
        Returns:
            pd.DataFrame: DataFrame containing player information
        """
        query = self._load_query('players')
        variables = {}
        if title_id:  # Only add title filter if title_id is specified
            variables['filter'] = {'titleId': title_id}
        
        try:
            # Get all player nodes, with optional limit
            players = self._execute_paginated_query(query, self.central_client, variables, limit=limit)
            
            # Process players into a list of dictionaries
            processed_players = []
            for player in players:
                player_data = {
                    'id': player.get('id'),
                    'nickname': player.get('nickname'),
                    'title': player.get('title', {}).get('name'),
                    'team_id': player.get('team', {}).get('id') if player.get('team') else None,
                    'team_name': player.get('team', {}).get('name') if player.get('team') else None,
                    'private': player.get('private', False)
                }
                processed_players.append(player_data)
            
            # Convert to DataFrame
            df = pd.DataFrame(processed_players)
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting player data: {str(e)}")
            raise


    def get_team_statistics(
        self,
        team_id: str,
        time_window: Optional[str] = "LAST_3_MONTHS",
        tournament_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive team statistics either by time window or tournament IDs.
        
        Args:
            team_id (str): The ID of the team
            time_window (str, optional): Time window for statistics (e.g., "LAST_3_MONTHS")
            tournament_ids (List[str], optional): List of tournament IDs to filter by
            
        Returns:
            Dict containing processed team statistics
        """
        query = self._load_query('team_statistics')
        
        # Build filter based on provided parameters
        filter_dict = {}
        if tournament_ids:
            filter_dict['tournamentIds'] = {'in': tournament_ids}
        elif time_window:
            filter_dict['timeWindow'] = time_window
            
        variables = {
            'teamId': team_id,
            'filter': filter_dict
        }
        
        try:
            result = self._execute_query(query, variables, self.stats_client)
            
            if not result or 'teamStatistics' not in result:
                return {}
                
            stats = result['teamStatistics']
            
            # Process and structure the statistics
            processed_stats = {
                'team_id': team_id,
                'series_count': stats['series']['count'],
                'game_count': stats['game']['count'],
                'kills': {
                    'total': stats['series']['kills']['sum'],
                    'avg_per_series': stats['series']['kills']['avg'],
                    'max_in_series': stats['series']['kills']['max'],
                    'min_in_series': stats['series']['kills']['min']
                },
                'wins': {
                    'count': stats['game']['wins']['count'],
                    'percentage': stats['game']['wins']['percentage'],
                    'current_streak': stats['game']['wins']['streak']['current'],
                    'max_streak': stats['game']['wins']['streak']['max']
                }
            }
            
            # Process segment statistics
            if 'segment' in stats:
                processed_stats['segments'] = []
                for segment in stats['segment']:
                    segment_data = {
                        'type': segment['type'],
                        'count': segment['count'],
                        'deaths': {
                            'total': segment['deaths']['sum'],
                            'avg': segment['deaths']['avg'],
                            'max': segment['deaths']['max'],
                            'min': segment['deaths']['min']
                        }
                    }
                    processed_stats['segments'].append(segment_data)
            
            return processed_stats
            
        except Exception as e:
            self.logger.error(f"Error fetching team statistics for team {team_id}: {str(e)}")
            return {}

    def get_player_statistics(
        self,
        player_id: str,
        time_window: Optional[str] = "LAST_3_MONTHS",
        tournament_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive player statistics either by time window or tournament IDs.
        
        Args:
            player_id (str): The ID of the player
            time_window (str, optional): Time window for statistics
            tournament_ids (List[str], optional): List of tournament IDs to filter by
            
        Returns:
            Dict containing processed player statistics
        """
        query = self._load_query('player_statistics')
        
        # Build filter based on provided parameters
        filter_dict = {}
        if tournament_ids:
            filter_dict['tournamentIds'] = {'in': tournament_ids}
        elif time_window:
            filter_dict['timeWindow'] = time_window
            
        variables = {
            'playerId': player_id,
            'filter': filter_dict
        }
        
        try:
            result = self._execute_query(query, variables, self.stats_client)
            
            if not result or 'playerStatistics' not in result:
                return {}
                
            stats = result['playerStatistics']
            
            # Process and structure the statistics
            processed_stats = {
                'player_id': player_id,
                'series_count': stats['series']['count'],
                'game_count': stats['game']['count'],
                'performance': {
                    'kills': {
                        'total': stats['series']['kills']['sum'],
                        'avg_per_series': stats['series']['kills']['avg'],
                        'max_in_series': stats['series']['kills']['max'],
                        'min_in_series': stats['series']['kills']['min']
                    },
                    'wins': {
                        'count': stats['game']['wins']['count'],
                        'percentage': stats['game']['wins']['percentage'],
                        'current_streak': stats['game']['wins']['streak']['current'],
                        'max_streak': stats['game']['wins']['streak']['max']
                    }
                }
            }
            
            # Add segment statistics if available
            if 'segment' in stats:
                processed_stats['segments'] = []
                for segment in stats['segment']:
                    segment_data = {
                        'type': segment['type'],
                        'count': segment['count'],
                        'deaths': {
                            'total': segment['deaths']['sum'],
                            'avg': segment['deaths']['avg'],
                            'max': segment['deaths']['max'],
                            'min': segment['deaths']['min']
                        }
                    }
                    processed_stats['segments'].append(segment_data)
            
            return processed_stats
            
        except Exception as e:
            self.logger.error(f"Error fetching player statistics for player {player_id}: {str(e)}")
            return {}

    def get_bulk_team_statistics(self, team_ids: List[str]) -> pd.DataFrame:
        """
        Get statistics for multiple teams and return as a DataFrame.
        
        Args:
            team_ids (List[str]): List of team IDs to fetch statistics for
            
        Returns:
            pd.DataFrame: DataFrame containing statistics for all teams
        """
        all_stats = []
        for team_id in team_ids:
            stats = self.get_team_statistics(team_id)
            if stats:
                # Flatten the nested dictionary for DataFrame compatibility
                flat_stats = {
                    'team_id': stats['team_id'],
                    'series_count': stats['series_count'],
                    'game_count': stats['game_count'],
                    'total_kills': stats['kills']['total'],
                    'avg_kills_per_series': stats['kills']['avg_per_series'],
                    'win_count': stats['wins']['count'],
                    'win_percentage': stats['wins']['percentage'],
                    'current_win_streak': stats['wins']['current_streak'],
                    'max_win_streak': stats['wins']['max_streak']
                }
                all_stats.append(flat_stats)
                
        return pd.DataFrame(all_stats)

    def get_bulk_player_statistics(self, player_ids: List[str]) -> pd.DataFrame:
        """
        Get statistics for multiple players and return as a DataFrame.
        
        Args:
            player_ids (List[str]): List of player IDs to fetch statistics for
            
        Returns:
            pd.DataFrame: DataFrame containing statistics for all players
        """
        all_stats = []
        for player_id in player_ids:
            stats = self.get_player_statistics(player_id)
            if stats:
                # Flatten the nested dictionary for DataFrame compatibility
                flat_stats = {
                    'player_id': stats['player_id'],
                    'series_count': stats['series_count'],
                    'game_count': stats['game_count'],
                    'total_kills': stats['performance']['kills']['total'],
                    'avg_kills_per_series': stats['performance']['kills']['avg_per_series'],
                    'win_count': stats['performance']['wins']['count'],
                    'win_percentage': stats['performance']['wins']['percentage'],
                    'current_win_streak': stats['performance']['wins']['current_streak'],
                    'max_win_streak': stats['performance']['wins']['max_streak']
                }
                all_stats.append(flat_stats)
                
        return pd.DataFrame(all_stats)
    
    def get_comprehensive_player_stats(self, player_id: str) -> Dict[str, Any]:
        """
        Get detailed player statistics including advanced metrics and game-specific stats.
        
        Args:
            player_id (str): The ID of the player
        
        Returns:
            Dict containing detailed player statistics
        """
        query = self._load_query('comprehensive_player_stats')
        
        variables = {
            'playerId': player_id,
            'filter': {
                'timeWindow': 'LAST_3_MONTHS'
            }
        }
        
        try:
            result = self._execute_query(query, variables, self.stats_client)
            
            if not result or 'playerStatistics' not in result:
                return {}
                
            stats = result['playerStatistics']
            game_stats = stats.get('game', {})
            
            # Base statistics structure
            processed_stats = {
                'player_id': player_id,
                'general': {
                    'series_played': stats['series']['count'],
                    'games_played': game_stats.get('count', 0)
                },
                'combat': {
                    'kills': {
                        'total': stats['series']['kills']['sum'],
                        'average': stats['series']['kills']['avg'],
                        'best': stats['series']['kills']['max'],
                        'worst': stats['series']['kills']['min']
                    },
                    'deaths': {
                        'total': stats['series']['deaths']['sum'],
                        'average': stats['series']['deaths']['avg'],
                        'worst': stats['series']['deaths']['max'],
                        'best': stats['series']['deaths']['min']
                    },
                    'first_kills': {
                        'total': stats['series'].get('firstKill', [{}])[0].get('count', 0),
                        'percentage': stats['series'].get('firstKill', [{}])[0].get('percentage', 0)
                    }
                },
                'performance': {
                    'wins': {
                        'count': game_stats.get('won', [{}])[0].get('count', 0),
                        'percentage': game_stats.get('won', [{}])[0].get('percentage', 0),
                        'current_streak': game_stats.get('won', [{}])[0].get('streak', {}).get('current', 0),
                        'max_streak': game_stats.get('won', [{}])[0].get('streak', {}).get('max', 0)
                    }
                },
                'economy': {
                    'net_worth': {
                        'average': game_stats.get('netWorth', {}).get('avg', 0),
                        'max': game_stats.get('netWorth', {}).get('max', 0)
                    },
                    'inventory_value': {
                        'average': game_stats.get('inventoryValue', {}).get('avg', 0),
                        'max': game_stats.get('inventoryValue', {}).get('max', 0)
                    }
                }
            }
            
            # Add game-specific statistics
            if 'damageDealt' in game_stats:  # CS2, CSGO, R6
                processed_stats['combat']['damage'] = {
                    'total': game_stats['damageDealt'].get('sum', 0),
                    'average': game_stats['damageDealt'].get('avg', 0),
                    'max': game_stats['damageDealt'].get('max', 0)
                }
            
            # Add MOBA-specific stats (DOTA, LOL)
            if 'experiencePoints' in game_stats:
                processed_stats['progression'] = {
                    'experience': {
                        'total': game_stats['experiencePoints'].get('sum', 0),
                        'average': game_stats['experiencePoints'].get('avg', 0),
                        'max': game_stats['experiencePoints'].get('max', 0)
                    }
                }
                
            # Add LOL-specific stats
            if 'totalMoneyEarned' in game_stats:
                processed_stats['economy']['total_earned'] = {
                    'total': game_stats['totalMoneyEarned'].get('sum', 0),
                    'average': game_stats['totalMoneyEarned'].get('avg', 0),
                    'max': game_stats['totalMoneyEarned'].get('max', 0)
                }
            
            # Add R6-specific stats
            if 'healingDealt' in game_stats:
                processed_stats['support'] = {
                    'healing_dealt': {
                        'total': game_stats['healingDealt'].get('sum', 0),
                        'average': game_stats['healingDealt'].get('avg', 0)
                    },
                    'healing_received': {
                        'total': game_stats['healingReceived'].get('sum', 0),
                        'average': game_stats['healingReceived'].get('avg', 0)
                    }
                }
            
            # Process unit kills if available
            if 'unitKills' in game_stats:
                processed_stats['combat']['unit_kills'] = {
                    unit['unitName']: {
                        'count': unit['count'].get('sum', 0),
                        'average': unit['count'].get('avg', 0)
                    }
                    for unit in game_stats.get('unitKills', [])
                }
            
            # Segment processing
            if 'segment' in stats:
                processed_stats['segments'] = []
                for segment in stats['segment']:
                    segment_data = {
                        'type': segment['type'],
                        'count': segment['count'],
                        'combat': {
                            'kills': {
                                'total': segment['kills']['sum'],
                                'average': segment['kills']['avg']
                            },
                            'deaths': {
                                'total': segment['deaths']['sum'],
                                'average': segment['deaths']['avg']
                            }
                        },
                        'objectives': [],
                        'win_rate': segment['won'][0]['percentage'] if segment.get('won') else 0
                    }
                    
                    if 'objectives' in segment:
                        for objective in segment['objectives']:
                            obj_data = {
                                'type': objective['type'],
                                'completions': objective['completionCount']['sum'],
                                'avg_completions': objective['completionCount']['avg'],
                                'first_completions': {
                                    'count': objective.get('completedFirst', [{}])[0].get('count', 0),
                                    'percentage': objective.get('completedFirst', [{}])[0].get('percentage', 0)
                                }
                            }
                            segment_data['objectives'].append(obj_data)
                    
                    processed_stats['segments'].append(segment_data)
            
            return processed_stats
                
        except Exception as e:
            self.logger.error(f"Error fetching comprehensive stats for player {player_id}: {str(e)}")
            return {}

    def collect_bulk_comprehensive_stats(self, 
                                    player_ids: Optional[List[str]] = None,
                                    batch_size: int = 50,
                                    save_intermediate: bool = True) -> pd.DataFrame:
        """
        Collect comprehensive statistics for multiple players with batching and save points.
        
        Args:
            player_ids: List of player IDs to collect stats for. If None, gets all players.
            batch_size: Number of players to process in each batch
            save_intermediate: Whether to save intermediate results
            
        Returns:
            DataFrame containing comprehensive statistics for all processed players
        """
        try:
            # If no player_ids provided, get all players
            if player_ids is None:
                players_df = self.get_players()
                player_ids = players_df['id'].tolist()
            
            total_players = len(player_ids)
            self.logger.info(f"Preparing to collect stats for {total_players} players")
            
            all_stats = []
            
            # Process in batches
            for i in range(0, total_players, batch_size):
                batch_ids = player_ids[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1} of {(total_players + batch_size - 1)//batch_size}")
                
                batch_stats = []
                for player_id in batch_ids:
                    try:
                        # Use your existing rate limiter
                        self.rate_limiter.wait()
                        
                        stats = self.get_comprehensive_player_stats(player_id)
                        if stats:
                            batch_stats.append(stats)
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing player {player_id}: {str(e)}")
                        continue
                
                all_stats.extend(batch_stats)
                
                # Save intermediate results if requested
                if save_intermediate and batch_stats:
                    batch_df = pd.DataFrame(batch_stats)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    batch_df.to_csv(f'player_stats_batch_{i//batch_size}_{timestamp}.csv', index=False)
                    
                    self.logger.info(f"Saved batch {i//batch_size + 1} results - {len(batch_stats)} players processed")
            
            # Create final DataFrame
            if all_stats:
                final_df = pd.DataFrame(all_stats)
                
                # Add some derived statistics
                if 'combat' in final_df.columns:
                    final_df['kd_ratio'] = final_df['combat'].apply(
                        lambda x: x['kills']['total'] / x['deaths']['total'] 
                        if x['deaths']['total'] > 0 else x['kills']['total']
                    )
                    
                    final_df['avg_damage_per_game'] = final_df.apply(
                        lambda x: x['combat']['damage']['total'] / x['general']['games_played']
                        if x['general']['games_played'] > 0 else 0,
                        axis=1
                    )
                
                return final_df
            else:
                self.logger.warning("No valid statistics were collected")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error in bulk collection: {str(e)}")
            raise

    def get_players_by_ids(self, player_ids: List[str]) -> pd.DataFrame:
        """
        Fetch player data for a list of player IDs using the central-data API.
        
        Args:
            player_ids (List[str]): List of player IDs to fetch data for
            
        Returns:
            pd.DataFrame: DataFrame containing player data
        """
        try:
            # Execute query with player IDs
            variables = {
                "playerIds": player_ids
            }
            
            result = self._execute_query(
                self._load_query('players_by_ids'),
                variables,
                self.central_client  # Make sure to use central-data client
            )
            
            if not result or 'players' not in result or 'edges' not in result['players']:
                logging.error("No player data returned from API")
                return pd.DataFrame()
                
            players_data = []
            
            for edge in result['players']['edges']:
                player = edge['node']
                player_info = {
                    'player_id': player['id'],
                    'nickname': player['nickname'],
                    'title_id': player['title']['id'] if player.get('title') else None,
                    'title_name': player['title']['name'] if player.get('title') else None,
                    'title_short_name': player['title']['shortName'] if player.get('title') else None,
                    'team_id': player['team']['id'] if player.get('team') else None,
                    'team_name': player['team']['name'] if player.get('team') else None,
                    'team_short_name': player['team']['shortName'] if player.get('team') else None,
                    'private': player.get('private', False),
                }
                
                # Add external links as separate columns
                external_links = {link['provider']: link['id'] for link in player.get('externalLinks', [])}
                player_info.update(external_links)
                
                players_data.append(player_info)
            
            return pd.DataFrame(players_data)
            
        except Exception as e:
            logging.error(f"Error fetching player data: {str(e)}")
            raise