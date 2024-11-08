from time import time, sleep

class RateLimiter:
    """Rate limiter for API requests"""
    def __init__(self, requests_per_minute: int = 20):
        self.interval = 60.0 / requests_per_minute
        self.last_request = 0.0

    def wait(self) -> None:
        """Wait if necessary to respect rate limits"""
        elapsed = time() - self.last_request
        if elapsed < self.interval:
            sleep(self.interval - elapsed)
        self.last_request = time()
