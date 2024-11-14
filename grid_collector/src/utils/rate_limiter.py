from time import time, sleep

class RateLimiter:
    """Enhanced rate limiter for API requests with burst handling"""
    def __init__(self, 
                 requests_per_minute: int = 20,
                 burst_limit: int = 5,
                 burst_cooldown: float = 10.0):
        self.interval = 60.0 / requests_per_minute
        self.burst_limit = burst_limit
        self.burst_cooldown = burst_cooldown
        self.last_request = 0.0
        self.requests_in_window = 0
        self.window_start = time()

    def wait(self) -> None:
        """Wait if necessary to respect rate limits"""
        current_time = time()
        
        # Reset burst window if cooldown has passed
        if current_time - self.window_start > self.burst_cooldown:
            self.requests_in_window = 0
            self.window_start = current_time

        # Check burst limit
        if self.requests_in_window >= self.burst_limit:
            sleep_time = self.burst_cooldown - (current_time - self.window_start)
            if sleep_time > 0:
                sleep(sleep_time)
            self.requests_in_window = 0
            self.window_start = time()

        # Normal rate limiting
        elapsed = current_time - self.last_request
        if elapsed < self.interval:
            sleep(self.interval - elapsed)
        
        self.last_request = time()
        self.requests_in_window += 1

    def reset(self) -> None:
        """Reset the rate limiter state"""
        self.last_request = 0.0
        self.requests_in_window = 0
        self.window_start = time()