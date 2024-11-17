import os
import sys
import logging
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from dotenv import load_dotenv

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grid_api_test.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv(os.path.join(project_root, '.env'))
api_key = os.getenv('GRID_API_KEY')

if not api_key:
    raise ValueError("GRID_API_KEY not found in environment variables")

def create_client():
    """Create a GraphQL client"""
    transport = RequestsHTTPTransport(
        url='https://api-op.grid.gg/central-data/graphql',
        headers={'x-api-key': api_key},
        retries=3
    )
    return Client(transport=transport, fetch_schema_from_transport=True)

def test_tournament_query():
    """Test a single tournament query"""
    # Create the query
    query = """
    query GetTournaments($first: Int!, $after: Cursor) {
        tournaments(
            first: $first,
            after: $after
        ) {
            pageInfo {
                hasNextPage
                endCursor
            }
            edges {
                cursor
                node {
                    id
                    name
                    nameShortened
                    startDate
                    endDate
                    private
                    titles {
                        id
                        name
                    }
                }
            }
            totalCount
        }
    }
    """
    
    try:
        # Execute query
        client = create_client()
        result = client.execute(
            gql(query),
            variable_values={'first': 1, 'after': None}
        )
        
        # Log the full response
        logging.info("Raw API Response:")
        logging.info(result)
        
        # Try to access the data
        tournaments = result.get('tournaments', {})
        edges = tournaments.get('edges', [])
        
        if edges:
            first_tournament = edges[0].get('node', {})
            logging.info("\nFirst Tournament Data:")
            logging.info(first_tournament)
            
            # Test accessing specific fields
            logging.info("\nTesting field access:")
            logging.info(f"ID: {first_tournament.get('id')}")
            logging.info(f"Name: {first_tournament.get('name')}")
            logging.info(f"Titles: {first_tournament.get('titles', [])}")
        else:
            logging.warning("No tournaments found in response")
            
    except Exception as e:
        logging.error(f"Error testing tournament query: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.info("Starting Grid API test")
    try:
        test_tournament_query()
        logging.info("Test completed successfully")
    except Exception as e:
        logging.error("Test failed", exc_info=True)