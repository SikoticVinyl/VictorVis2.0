from typing import Optional
import logging

logger = logging.getLogger(__name__)

class GridAPIError(Exception):
    """Custom exception for GRID API errors"""
    def __init__(self, message: str, error_type: str, error_detail: Optional[str] = None):
        self.error_type = error_type
        self.error_detail = error_detail
        super().__init__(message)

def handle_api_error(error: Exception) -> GridAPIError:
    """Convert API errors to GridAPIError"""
    if hasattr(error, 'errors') and error.errors:
        error_info = error.errors[0].get('extensions', {})
        return GridAPIError(
            str(error),
            error_info.get('errorType', 'UNKNOWN'),
            error_info.get('errorDetail')
        )
    return GridAPIError(str(error), 'UNKNOWN')