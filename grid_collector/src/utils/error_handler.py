from typing import Optional
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class GridErrorType(Enum):
    UNKNOWN = "UNKNOWN"
    INTERNAL = "INTERNAL"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHENTICATED = "UNAUTHENTICATED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    BAD_REQUEST = "BAD_REQUEST"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED_PRECONDITION = "FAILED_PRECONDITION"

class GridErrorDetail(Enum):
    UNKNOWN = "UNKNOWN"
    FIELD_NOT_FOUND = "FIELD_NOT_FOUND"
    INVALID_CURSOR = "INVALID_CURSOR"
    UNIMPLEMENTED = "UNIMPLEMENTED"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    SERVICE_ERROR = "SERVICE_ERROR"
    THROTTLED_CPU = "THROTTLED_CPU"
    THROTTLED_CONCURRENCY = "THROTTLED_CONCURRENCY"
    ENHANCE_YOUR_CALM = "ENHANCE_YOUR_CALM"
    TCP_FAILURE = "TCP_FAILURE"
    MISSING_RESOURCE = "MISSING_RESOURCE"

class GridAPIError(Exception):
    """Custom exception for GRID API errors with enhanced error handling"""
    def __init__(self, 
                 message: str, 
                 error_type: GridErrorType, 
                 error_detail: Optional[GridErrorDetail] = None,
                 retry_after: Optional[int] = None):
        self.error_type = error_type
        self.error_detail = error_detail
        self.retry_after = retry_after
        self.message = message
        super().__init__(message)

    @property
    def should_retry(self) -> bool:
        """Determine if the error is retryable"""
        retryable_types = {
            GridErrorType.UNAVAILABLE,
            GridErrorType.INTERNAL
        }
        retryable_details = {
            GridErrorDetail.THROTTLED_CPU,
            GridErrorDetail.THROTTLED_CONCURRENCY,
            GridErrorDetail.ENHANCE_YOUR_CALM,
            GridErrorDetail.TCP_FAILURE,
            GridErrorDetail.DEADLINE_EXCEEDED
        }
        return (self.error_type in retryable_types or 
                (self.error_detail and self.error_detail in retryable_details))

def handle_api_error(error: Exception) -> GridAPIError:
    """Convert API errors to GridAPIError with enhanced error handling"""
    if hasattr(error, 'errors') and error.errors:
        error_info = error.errors[0].get('extensions', {})
        error_type = error_info.get('errorType', 'UNKNOWN')
        error_detail = error_info.get('errorDetail')
        retry_after = error_info.get('retryAfter')
        
        try:
            error_type_enum = GridErrorType[error_type]
            error_detail_enum = GridErrorDetail[error_detail] if error_detail else None
        except KeyError:
            error_type_enum = GridErrorType.UNKNOWN
            error_detail_enum = None
            
        return GridAPIError(
            str(error),
            error_type_enum,
            error_detail_enum,
            retry_after
        )
        
    return GridAPIError(str(error), GridErrorType.UNKNOWN)

def log_and_handle_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """Utility function to log and handle errors appropriately"""
    if isinstance(error, GridAPIError):
        if error.should_retry:
            logger.warning(
                f"{context} - Retryable error occurred: {error.message} "
                f"(Type: {error.error_type.value}, Detail: {error.error_detail.value if error.error_detail else 'None'})"
            )
        else:
            logger.error(
                f"{context} - Non-retryable error occurred: {error.message} "
                f"(Type: {error.error_type.value}, Detail: {error.error_detail.value if error.error_detail else 'None'})"
            )
    else:
        logger.error(f"{context} - Unexpected error occurred: {str(error)}")