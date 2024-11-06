import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
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
        """Execute a single query with rate limiting."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Wait for rate limit
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

    def get_players(self, limit: int = None, title_id: str = '28') -> pd.DataFrame:
        """Get players with specified fields, optionally limiting the number of players fetched."""
        query = self._load_query('players')
        variables = {}
        if title_id:
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
    
    def get_player_statistics(self, player_id: str, time_window: str = 'LAST_YEAR') -> Dict[str, Any]:
        """Get statistics for a player, including CS:GO-specific statistics."""
        query = self._load_query('player_statistics')
        variables = {
            'playerId': player_id,
            'filter': {'timeWindow': time_window}
        }

        try:
            result = self._execute_query(query, variables, self.stats_client)
            return result['playerStatistics']
        except Exception as e:
            self.logger.error(f"Error collecting statistics for player {player_id}: {str(e)}")
            return None  # Return None if there's an error to handle it gracefully