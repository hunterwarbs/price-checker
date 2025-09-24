"""
Smart Adaptive Rate Limiter for OpenRouter and Oxylabs APIs

This module provides intelligent rate limiting with:
- Exponential backoff on 429 errors
- Dynamic rate adjustment based on success/failure patterns
- Provider-specific error handling
- Circuit breaker pattern for persistent failures
- Metrics and monitoring
"""

import asyncio
import os
import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import random
import httpx

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    OPENROUTER = "openrouter"
    OXYLABS_PROXY = "oxylabs_proxy"      # For screenshot/browser automation
    OXYLABS_WEB_API = "oxylabs_web_api"  # For Google search API calls


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting per provider"""
    initial_requests_per_second: float
    max_requests_per_second: float
    min_requests_per_second: float
    backoff_multiplier: float = 2.0
    max_backoff_time: float = 300.0  # 5 minutes
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    success_rate_threshold: float = 0.8  # 80% success rate to increase limits


@dataclass
class RequestMetrics:
    """Metrics for tracking request performance"""
    total_requests: int = 0
    successful_requests: int = 0
    rate_limited_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_success_time: float = 0.0
    last_rate_limit_time: float = 0.0
    consecutive_failures: int = 0
    request_times: deque = field(default_factory=lambda: deque(maxlen=100))


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit breaker active
    HALF_OPEN = "half_open"  # Testing if service recovered


class SmartRateLimiter:
    """Smart adaptive rate limiter with circuit breaker and dynamic adjustment"""
    
    def __init__(self, provider: ProviderType, config: RateLimitConfig):
        self.provider = provider
        self.config = config
        self.metrics = RequestMetrics()
        
        # Rate limiting state
        self.current_rate = config.initial_requests_per_second
        self.request_queue = deque()
        self.last_request_time = 0.0
        
        # Circuit breaker state
        self.circuit_state = CircuitState.CLOSED
        self.circuit_open_time = 0.0
        
        # Adaptive adjustment
        self.adjustment_window_size = 50  # Evaluate every 50 requests
        self.last_adjustment_time = time.time()
        
        # Provider-specific settings
        self._configure_provider_specifics()
        
        logger.info(f"Smart rate limiter initialized for {provider.value} - Initial rate: {self.current_rate} RPS")
    
    def _configure_provider_specifics(self):
        """Configure provider-specific settings based on research"""
        if self.provider == ProviderType.OPENROUTER:
            # OpenRouter: 20 RPM free, varies by model, has Retry-After headers
            self.error_codes = [429, 402]  # 429 rate limit, 402 insufficient credits
            self.respect_retry_after = True
            

        elif self.provider == ProviderType.OXYLABS_PROXY:
            # Oxylabs Proxy: For screenshots/browser automation, more conservative
            self.error_codes = [429, 407, 408, 502, 503]  # Including proxy errors
            self.respect_retry_after = True
            
        elif self.provider == ProviderType.OXYLABS_WEB_API:
            # Oxylabs Web API: For search requests, can handle higher rates
            self.error_codes = [429, 402, 500, 502, 503]  # API-specific errors  
            self.respect_retry_after = True
    
    async def execute_with_rate_limit(
        self, 
        request_func: Callable[[], Awaitable[Any]],
        max_retries: int = 3
    ) -> Any:
        """
        Execute a request with smart rate limiting and retries
        
        Args:
            request_func: Async function that makes the actual request
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response from the request function
            
        Raises:
            Exception: If all retries are exhausted or circuit breaker is open
        """
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise Exception(f"Circuit breaker OPEN for {self.provider.value} - service temporarily unavailable")
        
        retries = 0
        last_exception = None
        
        while retries <= max_retries:
            try:
                # Wait for rate limit
                await self._wait_for_rate_limit()
                
                # Execute request with timing
                start_time = time.time()
                result = await request_func()
                response_time = time.time() - start_time
                
                # Record success
                self._record_success(response_time)
                
                # Adjust rate limits based on success
                self._adjust_rate_limits()
                
                return result
                
            except Exception as e:
                last_exception = e
                response_time = time.time() - start_time if 'start_time' in locals() else 0
                
                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    logger.warning(f"{self.provider.value} rate limited - attempt {retries + 1}/{max_retries + 1}")
                    
                    # Record rate limit
                    self._record_rate_limit(e)
                    
                    # Calculate backoff time
                    backoff_time = self._calculate_backoff_time(retries, e)
                    
                    # Reduce rate limits
                    self._reduce_rate_limits()
                    
                    if retries < max_retries:
                        logger.info(f"Backing off for {backoff_time:.2f} seconds")
                        await asyncio.sleep(backoff_time)
                        retries += 1
                        continue
                else:
                    # Non-rate-limit error
                    self._record_failure(response_time)
                    
                    # Check if we should trigger circuit breaker
                    if self._should_open_circuit():
                        self._open_circuit()
                    
                    raise e
        
        # All retries exhausted
        self._record_failure(0)
        
        if self._should_open_circuit():
            self._open_circuit()
            
        raise last_exception
    
    async def _wait_for_rate_limit(self):
        """Wait if necessary to respect current rate limit"""
        now = time.time()
        
        # Calculate time since last request
        time_since_last = now - self.last_request_time
        
        # Calculate minimum time between requests
        min_interval = 1.0 / self.current_rate
        
        # Wait if needed
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limiting error"""
        if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            return error.response.status_code in self.error_codes
        
        # Check for specific error messages
        error_str = str(error).lower()
        rate_limit_indicators = [
            'rate limit', 'too many requests', '429', 'quota exceeded',
            'request rate', 'throttled', 'rate exceeded'
        ]
        
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def _calculate_backoff_time(self, retry_count: int, error: Exception) -> float:
        """Calculate backoff time with exponential backoff and jitter"""
        
        # Check for Retry-After header if we respect it
        if self.respect_retry_after and hasattr(error, 'response'):
            retry_after = getattr(error.response, 'headers', {}).get('Retry-After')
            if retry_after:
                try:
                    return float(retry_after) + random.uniform(0.1, 0.5)
                except (ValueError, TypeError):
                    pass
        
        # Exponential backoff with jitter
        base_delay = min(
            self.config.backoff_multiplier ** retry_count,
            self.config.max_backoff_time
        )
        
        # Add jitter (±25%)
        jitter = base_delay * 0.25 * (2 * random.random() - 1)
        
        return max(0.1, base_delay + jitter)
    
    def _record_success(self, response_time: float):
        """Record a successful request"""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = time.time()
        self.metrics.consecutive_failures = 0
        
        # Update average response time
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = response_time
        else:
            # Exponential moving average
            self.metrics.average_response_time = (
                0.9 * self.metrics.average_response_time + 0.1 * response_time
            )
        
        self.metrics.request_times.append(time.time())
    
    def _record_rate_limit(self, error: Exception):
        """Record a rate limited request"""
        self.metrics.total_requests += 1
        self.metrics.rate_limited_requests += 1
        self.metrics.last_rate_limit_time = time.time()
        self.metrics.consecutive_failures += 1
    
    def _record_failure(self, response_time: float):
        """Record a failed request"""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.consecutive_failures += 1
        
        if response_time > 0:
            # Update average response time for failures too
            if self.metrics.average_response_time == 0:
                self.metrics.average_response_time = response_time
            else:
                self.metrics.average_response_time = (
                    0.9 * self.metrics.average_response_time + 0.1 * response_time
                )
    
    def _adjust_rate_limits(self):
        """Dynamically adjust rate limits based on performance"""
        if self.metrics.total_requests % self.adjustment_window_size != 0:
            return
        
        now = time.time()
        if now - self.last_adjustment_time < 30:  # Don't adjust more than once per 30 seconds
            return
        
        self.last_adjustment_time = now
        
        # Calculate success rate over recent requests
        recent_window = min(self.adjustment_window_size, self.metrics.total_requests)
        success_rate = self.metrics.successful_requests / max(1, self.metrics.total_requests)
        
        logger.debug(f"Adjusting rate limits - Success rate: {success_rate:.2%}, Current rate: {self.current_rate:.2f} RPS")
        
        if success_rate >= self.config.success_rate_threshold:
            # High success rate - increase rate limit
            new_rate = min(
                self.current_rate * 1.1,  # 10% increase
                self.config.max_requests_per_second
            )
            if new_rate > self.current_rate:
                self.current_rate = new_rate
                logger.info(f"Increased rate limit to {self.current_rate:.2f} RPS for {self.provider.value}")
        
        elif success_rate < 0.5:  # Less than 50% success rate
            # Low success rate - decrease rate limit
            new_rate = max(
                self.current_rate * 0.8,  # 20% decrease
                self.config.min_requests_per_second
            )
            if new_rate < self.current_rate:
                self.current_rate = new_rate
                logger.info(f"Decreased rate limit to {self.current_rate:.2f} RPS for {self.provider.value}")
    
    def _reduce_rate_limits(self):
        """Reduce rate limits after hitting rate limits"""
        new_rate = max(
            self.current_rate * 0.7,  # 30% decrease
            self.config.min_requests_per_second
        )
        
        if new_rate < self.current_rate:
            self.current_rate = new_rate
            logger.info(f"Reduced rate limit to {self.current_rate:.2f} RPS for {self.provider.value} due to rate limiting")
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests"""
        now = time.time()
        
        if self.circuit_state == CircuitState.CLOSED:
            return True
        
        elif self.circuit_state == CircuitState.OPEN:
            # Check if timeout has passed
            if now - self.circuit_open_time >= self.config.circuit_breaker_timeout:
                logger.info(f"Circuit breaker transitioning to HALF_OPEN for {self.provider.value}")
                self.circuit_state = CircuitState.HALF_OPEN
                return True
            return False
        
        elif self.circuit_state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit breaker should be opened"""
        return (
            self.metrics.consecutive_failures >= self.config.circuit_breaker_threshold and
            self.circuit_state == CircuitState.CLOSED
        )
    
    def _open_circuit(self):
        """Open the circuit breaker"""
        self.circuit_state = CircuitState.OPEN
        self.circuit_open_time = time.time()
        logger.error(f"Circuit breaker OPENED for {self.provider.value} - too many consecutive failures")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics and status"""
        success_rate = (
            self.metrics.successful_requests / max(1, self.metrics.total_requests)
        ) * 100
        
        return {
            "provider": self.provider.value,
            "circuit_state": self.circuit_state.value,
            "current_rate_rps": round(self.current_rate, 2),
            "total_requests": self.metrics.total_requests,
            "success_rate": round(success_rate, 2),
            "rate_limited_requests": self.metrics.rate_limited_requests,
            "failed_requests": self.metrics.failed_requests,
            "consecutive_failures": self.metrics.consecutive_failures,
            "average_response_time": round(self.metrics.average_response_time, 3),
            "last_success_time": self.metrics.last_success_time,
            "last_rate_limit_time": self.metrics.last_rate_limit_time
        }


class GlobalRateLimitManager:
    """Global manager for all rate limiters"""
    
    def __init__(self):
        self.limiters: Dict[ProviderType, SmartRateLimiter] = {}
        self._initialize_limiters()
    
    def _initialize_limiters(self):
        """Initialize rate limiters for each provider with research-based configs"""
        
        # OpenRouter configuration (expose via env vars for flexibility)
        openrouter_config = RateLimitConfig(
            initial_requests_per_second=float(os.getenv("SMART_OPENROUTER_INITIAL_RPS", "10.0")),  # Default 10 RPS
            max_requests_per_second=float(os.getenv("SMART_OPENROUTER_MAX_RPS", "20.0")),         # Allow bursting up to 20 RPS if stable
            min_requests_per_second=0.5,       # Fallback
            backoff_multiplier=2.0,
            max_backoff_time=300.0,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60.0
        )
        

        
        # Oxylabs Proxy configuration (for screenshots/browser automation - more conservative)
        oxylabs_proxy_config = RateLimitConfig(
            initial_requests_per_second=float(os.getenv("SMART_OXYLABS_PROXY_INITIAL_RPS", "10.0")),  # Conservative for browser automation
            max_requests_per_second=float(os.getenv("SMART_OXYLABS_PROXY_MAX_RPS", "25.0")),          # Moderate max for stability
            min_requests_per_second=1.0,       # Conservative fallback
            backoff_multiplier=2.0,            # More aggressive backoff for stability
            max_backoff_time=120.0,            # Longer backoff for browser issues
            circuit_breaker_threshold=5,       # Lower threshold for browser stability
            circuit_breaker_timeout=60.0       # Standard timeout
        )
        
        # Oxylabs Web API configuration (for Google search - maxed out at 50 req/s)
        oxylabs_web_api_config = RateLimitConfig(
            initial_requests_per_second=float(os.getenv("SMART_OXYLABS_WEB_API_INITIAL_RPS", "45.0")),  # Start high for search API
            max_requests_per_second=float(os.getenv("SMART_OXYLABS_WEB_API_MAX_RPS", "50.0")),         # User requested maximum
            min_requests_per_second=5.0,       # Higher minimum for API calls
            backoff_multiplier=1.2,            # Gentle backoff for API
            max_backoff_time=30.0,             # Quick recovery for search
            circuit_breaker_threshold=15,      # Higher threshold for API
            circuit_breaker_timeout=20.0       # Quick circuit breaker reset
        )
        
        # Initialize limiters
        self.limiters[ProviderType.OPENROUTER] = SmartRateLimiter(ProviderType.OPENROUTER, openrouter_config)
        self.limiters[ProviderType.OXYLABS_PROXY] = SmartRateLimiter(ProviderType.OXYLABS_PROXY, oxylabs_proxy_config)
        self.limiters[ProviderType.OXYLABS_WEB_API] = SmartRateLimiter(ProviderType.OXYLABS_WEB_API, oxylabs_web_api_config)
        
        logger.info("Global rate limit manager initialized with all providers")
    
    def get_limiter(self, provider: ProviderType) -> SmartRateLimiter:
        """Get rate limiter for specific provider"""
        return self.limiters[provider]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all providers"""
        return {
            provider.value: limiter.get_metrics()
            for provider, limiter in self.limiters.items()
        }
    
    async def execute_openrouter_request(self, request_func: Callable[[], Awaitable[Any]]) -> Any:
        """Execute OpenRouter request with smart rate limiting"""
        return await self.limiters[ProviderType.OPENROUTER].execute_with_rate_limit(request_func)
    

    
    async def execute_oxylabs_proxy_request(self, request_func: Callable[[], Awaitable[Any]]) -> Any:
        """Execute Oxylabs proxy request (for screenshots/browser automation) with smart rate limiting"""
        return await self.limiters[ProviderType.OXYLABS_PROXY].execute_with_rate_limit(request_func)
    
    async def execute_oxylabs_web_api_request(self, request_func: Callable[[], Awaitable[Any]]) -> Any:
        """Execute Oxylabs Web API request (for Google search) with smart rate limiting"""
        return await self.limiters[ProviderType.OXYLABS_WEB_API].execute_with_rate_limit(request_func)


# Global instance
global_rate_manager = GlobalRateLimitManager()