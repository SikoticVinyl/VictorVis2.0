import os
from typing import Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

from ..utils import RateLimiter, GridAPIError, handle_api_error
from ..models import Player, Match
from .base import BaseCollector

class GridCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = RateLimiter()
        
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
        """Load a GraphQL query from file using absolute path"""
        query_path = os.path.join(self.base_dir, 'queries', f'{filename}.graphql')
        print(f"Looking for query file at: {query_path}")
        print(f"File exists: {os.path.exists(query_path)}")
        
        try:
            with open(query_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Query file not found at: {query_path}\n"
                f"Base directory: {self.base_dir}\n"
                f"Available files in queries dir: {os.listdir(os.path.join(self.base_dir, 'queries'))}"
            )

    def _paginated_query(self, query: str, variables: Dict, client: Client) -> list:
        """Execute a paginated query and return all results"""
        results = []
        has_next_page = True
        after = None
        
        while has_next_page:
            self.rate_limiter.wait()
            try:
                current_vars = {**variables, 'after': after, 'first': 50}
                result = client.execute(gql(query), variable_values=current_vars)
                data = list(result.values())[0]
                results.extend(edge['node'] for edge in data['edges'])
                page_info = data['pageInfo']
                has_next_page = page_info['hasNextPage']
                after = page_info['endCursor'] if has_next_page else None
            except Exception as e:
                raise handle_api_error(e)
        return results

    def get_matches(self, days: int = 7) -> pd.DataFrame:
        """Get recent matches"""
        query = self._load_query('matches')
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        end_date = datetime.now().isoformat()
        
        try:
            self.rate_limiter.wait()
            result = self.central_client.execute(
                gql(query),
                variable_values={
                    'startDate': start_date,
                    'endDate': end_date,
                    'first': 50
                }
            )
            
            matches = []
            for edge in result['allSeries']['edges']:
                match = edge['node']
                if len(match['teams']) >= 2:
                    matches.append(Match(
                        id=match['id'],
                        start_time=datetime.fromisoformat(match['startTimeScheduled']),
                        tournament_name=match['tournament']['name'],
                        team1_id=match['teams'][0]['baseInfo']['id'],
                        team1_name=match['teams'][0]['baseInfo']['name'],
                        team2_id=match['teams'][1]['baseInfo']['id'],
                        team2_name=match['teams'][1]['baseInfo']['name'],
                        team1_score=match['teams'][0]['scoreAdvantage'],
                        team2_score=match['teams'][1]['scoreAdvantage']
                    ))
            
            return pd.DataFrame([vars(m) for m in matches])
            
        except Exception as e:
            raise handle_api_error(e)

    def get_player_stats(self, player_id: str, time_window: str = 'LAST_3_MONTHS') -> pd.DataFrame:
        """Get player statistics"""
        query = self._load_query('statistics')
        
        try:
            self.rate_limiter.wait()
            result = self.stats_client.execute(
                gql(query),
                variable_values={
                    'playerId': player_id,
                    'timeWindow': time_window
                }
            )
            return pd.json_normalize(result['playerStatistics'])
        except Exception as e:
            raise handle_api_error(e)
        
    def get_tournaments(self) -> pd.DataFrame:
        """Get all tournaments"""
        query = self._load_query('tournaments')
        results = []
        
        try:
            for edge in self._paginated_query(query, {}, self.central_client):
                tournament = edge['node']
                results.append({
                    'id': tournament['id'],
                    'name': tournament['name'],
                    'name_short': tournament['nameShortened']
                })
            
            return pd.DataFrame(results)
        except Exception as e:
            raise handle_api_error(e)

    def get_team_roster(self, team_id: str) -> pd.DataFrame:
        """Get roster for a specific team"""
        query = self._load_query('players')
        variables = {'teamId': team_id}
        players = []
        
        try:
            for edge in self._paginated_query(query, variables, self.central_client):
                player = edge['node']
                players.append({
                    'player_id': player['id'],
                    'nickname': player['nickname'],
                    'title': player['title']['name'] if player['title'] else None,
                    'team_id': team_id
                })
            
            return pd.DataFrame(players)
        except Exception as e:
            raise handle_api_error(e)

    def get_player_statistics(self, player_id: str, time_window: str = 'LAST_3_MONTHS') -> pd.DataFrame:
        """Get statistics for a specific player"""
        query = self._load_query('statistics')
        
        try:
            self.rate_limiter.wait()
            result = self.stats_client.execute(
                gql(query),
                variable_values={
                    'playerId': player_id,
                    'timeWindow': time_window
                }
            )
            return pd.json_normalize(result['playerStatistics'])
        except Exception as e:
            raise handle_api_error(e)

    def get_teams(self) -> pd.DataFrame:
        """Get all teams"""
        query = self._load_query('teams')
        teams = []
        
        try:
            for edge in self._paginated_query(query, {}, self.central_client):
                team = edge['node']
                teams.append({
                    'id': team['id'],
                    'name': team['name'],
                    'color_primary': team['colorPrimary'],
                    'color_secondary': team['colorSecondary'],
                    'logo_url': team['logoUrl']
                })
            
            return pd.DataFrame(teams)
        except Exception as e:
            raise handle_api_error(e)