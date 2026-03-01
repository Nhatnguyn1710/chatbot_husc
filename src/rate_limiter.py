
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Set
from datetime import datetime, timedelta
import json
import hashlib

@dataclass
class RateLimitConfig:
    """Configuration for Rate Limiter."""
    
    # Rate limits
    requests_per_minute: int = field(
        default_factory=lambda: int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))
    )
    requests_per_hour: int = field(
        default_factory=lambda: int(os.getenv("MAX_REQUESTS_PER_HOUR", "200"))
    )
    requests_per_day: int = field(
        default_factory=lambda: int(os.getenv("MAX_REQUESTS_PER_DAY", "1000"))
    )
    
    # Burst handling
    burst_limit: int = field(
        default_factory=lambda: int(os.getenv("BURST_LIMIT", "10"))
    )
    burst_window_seconds: int = 5
    
    # Cooldown
    cooldown_minutes: int = field(
        default_factory=lambda: int(os.getenv("COOLDOWN_MINUTES", "5"))
    )
    
    # Auto-block thresholds
    abuse_threshold_per_minute: int = field(
        default_factory=lambda: int(os.getenv("ABUSE_THRESHOLD", "100"))
    )
    block_duration_minutes: int = field(
        default_factory=lambda: int(os.getenv("BLOCK_DURATION_MINUTES", "60"))
    )
    
    # IP lists
    whitelisted_ips: Set[str] = field(default_factory=set)
    blacklisted_ips: Set[str] = field(default_factory=set)
    
    # Premium users (higher limits)
    premium_multiplier: float = 2.0
    
    # Cleanup
    cleanup_interval_seconds: int = 300  # 5 minutes
    max_entries_per_category: int = 10000
    
    def __post_init__(self):
        """Load IP lists from environment."""
        # Load whitelisted IPs
        whitelist = os.getenv("ALLOWED_IPS", "")
        if whitelist:
            self.whitelisted_ips = set(ip.strip() for ip in whitelist.split(",") if ip.strip())
        
        # Always whitelist localhost
        self.whitelisted_ips.add("127.0.0.1")
        self.whitelisted_ips.add("::1")
        self.whitelisted_ips.add("localhost")
        
        # Load blacklisted IPs
        blacklist = os.getenv("BLOCKED_IPS", "")
        if blacklist:
            self.blacklisted_ips = set(ip.strip() for ip in blacklist.split(",") if ip.strip())


# ==============================================================================
# RATE LIMIT RESULT
# ==============================================================================

@dataclass
class RateLimitResult:
    """Result from rate limit check."""
    allowed: bool
    message: str
    remaining: int = 0
    reset_time: Optional[float] = None
    retry_after: Optional[int] = None
    limit_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "allowed": self.allowed,
            "message": self.message,
            "remaining": self.remaining,
            "reset_time": self.reset_time,
            "retry_after": self.retry_after,
            "limit_type": self.limit_type
        }


# ==============================================================================
# SLIDING WINDOW COUNTER
# ==============================================================================

class SlidingWindowCounter:
    """
    Sliding window counter for rate limiting.
    More accurate than fixed window, less memory than sliding log.
    """
    
    def __init__(self, window_size_seconds: int):
        self.window_size = window_size_seconds
        self.current_window_start = 0
        self.current_count = 0
        self.previous_count = 0
        self._lock = threading.Lock()
    
    def increment(self, timestamp: Optional[float] = None) -> int:
        """
        Increment counter and return current count in window.
        
        Args:
            timestamp: Optional timestamp (default: now)
            
        Returns:
            Current count in sliding window
        """
        ts = timestamp or time.time()
        
        with self._lock:
            window_start = int(ts // self.window_size) * self.window_size
            
            if window_start > self.current_window_start:
                # New window
                if window_start == self.current_window_start + self.window_size:
                    # Just moved to next window
                    self.previous_count = self.current_count
                else:
                    # Skipped window(s)
                    self.previous_count = 0
                
                self.current_count = 0
                self.current_window_start = window_start
            
            self.current_count += 1
            
            # Calculate weighted count
            elapsed = ts - window_start
            weight = elapsed / self.window_size
            count = int(self.previous_count * (1 - weight) + self.current_count)
            
            return count
    
    def get_count(self, timestamp: Optional[float] = None) -> int:
        """Get current count without incrementing."""
        ts = timestamp or time.time()
        
        with self._lock:
            window_start = int(ts // self.window_size) * self.window_size
            
            if window_start > self.current_window_start + self.window_size:
                return 0
            
            if window_start == self.current_window_start:
                elapsed = ts - window_start
                weight = elapsed / self.window_size
                return int(self.previous_count * (1 - weight) + self.current_count)
            
            return 0
    
    def reset(self):
        """Reset counter."""
        with self._lock:
            self.current_window_start = 0
            self.current_count = 0
            self.previous_count = 0


# ==============================================================================
# RATE LIMITER
# ==============================================================================

class RateLimiter:
    """
    Rate limiter với multi-tier limits.
    
    Usage:
        limiter = RateLimiter()
        
        # Check before processing request
        result = limiter.check_rate_limit(ip="192.168.1.1", user_id="user123")
        
        if not result.allowed:
            return {"error": result.message}, 429
        
        # Process request...
    """
    
    _instance: Optional["RateLimiter"] = None
    
    def __new__(cls, config: Optional[RateLimitConfig] = None):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        if self._initialized:
            return
            
        self.config = config or RateLimitConfig()
        
        # Counters by IP
        self._ip_minute_counters: Dict[str, SlidingWindowCounter] = defaultdict(
            lambda: SlidingWindowCounter(60)
        )
        self._ip_hour_counters: Dict[str, SlidingWindowCounter] = defaultdict(
            lambda: SlidingWindowCounter(3600)
        )
        self._ip_day_counters: Dict[str, SlidingWindowCounter] = defaultdict(
            lambda: SlidingWindowCounter(86400)
        )
        
        # Counters by User ID
        self._user_minute_counters: Dict[str, SlidingWindowCounter] = defaultdict(
            lambda: SlidingWindowCounter(60)
        )
        self._user_hour_counters: Dict[str, SlidingWindowCounter] = defaultdict(
            lambda: SlidingWindowCounter(3600)
        )
        self._user_day_counters: Dict[str, SlidingWindowCounter] = defaultdict(
            lambda: SlidingWindowCounter(86400)
        )
        
        # Burst counters
        self._burst_counters: Dict[str, SlidingWindowCounter] = defaultdict(
            lambda: SlidingWindowCounter(self.config.burst_window_seconds)
        )
        
        # Cooldown tracking
        self._cooldowns: Dict[str, float] = {}  # key -> end_time
        
        # Auto-blocked IPs
        self._auto_blocked: Dict[str, float] = {}  # ip -> end_time
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "blocked_requests": 0,
            "rate_limited_requests": 0,
            "blocked_ips_count": 0
        }
        
        # Premium users
        self._premium_users: Set[str] = set()
        
        # Cleanup thread
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = False
        
        self._initialized = True
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self._cleanup_thread is not None:
            return
        
        def cleanup_loop():
            while not self._stop_cleanup:
                time.sleep(self.config.cleanup_interval_seconds)
                if not self._stop_cleanup:
                    self._cleanup_old_entries()
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_old_entries(self):
        """Remove old/expired entries to free memory."""
        now = time.time()
        
        with self._lock:
            # Cleanup expired auto-blocks
            expired_blocks = [
                ip for ip, end_time in self._auto_blocked.items()
                if end_time < now
            ]
            for ip in expired_blocks:
                del self._auto_blocked[ip]
            
            # Cleanup expired cooldowns
            expired_cooldowns = [
                key for key, end_time in self._cooldowns.items()
                if end_time < now
            ]
            for key in expired_cooldowns:
                del self._cooldowns[key]
            
            # Limit entries if too many
            for counter_dict in [
                self._ip_minute_counters,
                self._ip_hour_counters,
                self._ip_day_counters,
                self._user_minute_counters,
                self._user_hour_counters,
                self._user_day_counters
            ]:
                if len(counter_dict) > self.config.max_entries_per_category:
                    # Keep only most recent half
                    keys = list(counter_dict.keys())
                    for key in keys[:len(keys) // 2]:
                        del counter_dict[key]
    
    # ==========================================================================
    # MAIN RATE LIMIT CHECK
    # ==========================================================================
    
    def check_rate_limit(
        self,
        ip: Optional[str] = None,
        user_id: Optional[str] = None,
        is_premium: bool = False
    ) -> RateLimitResult:
        """
        Check if request should be allowed.
        
        Args:
            ip: Client IP address
            user_id: Optional user identifier
            is_premium: Whether user has premium status
            
        Returns:
            RateLimitResult with allow/deny decision
        """
        self._stats["total_requests"] += 1
        
        # Check blacklist first
        if ip and ip in self.config.blacklisted_ips:
            self._stats["blocked_requests"] += 1
            return RateLimitResult(
                allowed=False,
                message="IP đã bị chặn",
                limit_type="blacklist"
            )
        
        # Check auto-blocked
        if ip and ip in self._auto_blocked:
            if self._auto_blocked[ip] > time.time():
                self._stats["blocked_requests"] += 1
                retry_after = int(self._auto_blocked[ip] - time.time())
                return RateLimitResult(
                    allowed=False,
                    message=f"IP tạm thời bị chặn do hoạt động bất thường. Thử lại sau {retry_after}s",
                    retry_after=retry_after,
                    limit_type="auto_block"
                )
            else:
                # Expired, remove
                del self._auto_blocked[ip]
        
        # Check whitelist
        if ip and ip in self.config.whitelisted_ips:
            self._stats["allowed_requests"] += 1
            return RateLimitResult(
                allowed=True,
                message="OK (whitelisted)",
                remaining=-1  # Unlimited
            )
        
        # Check cooldown
        cooldown_key = f"{ip}:{user_id}" if ip and user_id else (ip or user_id)
        if cooldown_key and cooldown_key in self._cooldowns:
            if self._cooldowns[cooldown_key] > time.time():
                self._stats["rate_limited_requests"] += 1
                retry_after = int(self._cooldowns[cooldown_key] - time.time())
                return RateLimitResult(
                    allowed=False,
                    message=f"Bạn đang trong thời gian chờ. Thử lại sau {retry_after}s",
                    retry_after=retry_after,
                    limit_type="cooldown"
                )
            else:
                del self._cooldowns[cooldown_key]
        
        # Calculate limits (premium gets multiplier)
        multiplier = self.config.premium_multiplier if is_premium else 1.0
        limit_per_minute = int(self.config.requests_per_minute * multiplier)
        limit_per_hour = int(self.config.requests_per_hour * multiplier)
        limit_per_day = int(self.config.requests_per_day * multiplier)
        
        # Check burst limit
        if ip:
            burst_count = self._burst_counters[ip].get_count()
            if burst_count >= self.config.burst_limit:
                self._stats["rate_limited_requests"] += 1
                return RateLimitResult(
                    allowed=False,
                    message="Quá nhiều requests trong thời gian ngắn. Vui lòng chờ vài giây.",
                    retry_after=self.config.burst_window_seconds,
                    limit_type="burst"
                )
        
        # Check IP-based limits
        if ip:
            # Check for abuse (before incrementing)
            minute_count = self._ip_minute_counters[ip].get_count()
            if minute_count >= self.config.abuse_threshold_per_minute:
                # Auto-block this IP
                self._auto_block_ip(ip)
                self._stats["blocked_requests"] += 1
                return RateLimitResult(
                    allowed=False,
                    message="Phát hiện hoạt động bất thường. IP đã bị tạm chặn.",
                    retry_after=self.config.block_duration_minutes * 60,
                    limit_type="abuse_detected"
                )
            
            # Check minute limit
            minute_count = self._ip_minute_counters[ip].increment()
            if minute_count > limit_per_minute:
                self._set_cooldown(cooldown_key)
                self._stats["rate_limited_requests"] += 1
                return RateLimitResult(
                    allowed=False,
                    message=f"Đã vượt quá {limit_per_minute} requests/phút",
                    remaining=0,
                    retry_after=60,
                    limit_type="minute"
                )
            
            # Check hour limit
            hour_count = self._ip_hour_counters[ip].increment()
            if hour_count > limit_per_hour:
                self._set_cooldown(cooldown_key)
                self._stats["rate_limited_requests"] += 1
                return RateLimitResult(
                    allowed=False,
                    message=f"Đã vượt quá {limit_per_hour} requests/giờ",
                    remaining=0,
                    retry_after=300,
                    limit_type="hour"
                )
            
            # Check day limit
            day_count = self._ip_day_counters[ip].increment()
            if day_count > limit_per_day:
                self._stats["rate_limited_requests"] += 1
                return RateLimitResult(
                    allowed=False,
                    message=f"Đã vượt quá {limit_per_day} requests/ngày",
                    remaining=0,
                    retry_after=3600,
                    limit_type="day"
                )
            
            # Update burst counter
            self._burst_counters[ip].increment()
        
        # Check user-based limits (if user_id provided)
        if user_id:
            user_minute = self._user_minute_counters[user_id].increment()
            if user_minute > limit_per_minute:
                self._set_cooldown(cooldown_key)
                self._stats["rate_limited_requests"] += 1
                return RateLimitResult(
                    allowed=False,
                    message=f"Đã vượt quá {limit_per_minute} requests/phút cho user",
                    remaining=0,
                    retry_after=60,
                    limit_type="user_minute"
                )
            
            user_hour = self._user_hour_counters[user_id].increment()
            if user_hour > limit_per_hour:
                self._stats["rate_limited_requests"] += 1
                return RateLimitResult(
                    allowed=False,
                    message=f"Đã vượt quá {limit_per_hour} requests/giờ cho user",
                    remaining=0,
                    retry_after=300,
                    limit_type="user_hour"
                )
        
        # Calculate remaining
        remaining = limit_per_minute - (minute_count if ip else 0)
        
        self._stats["allowed_requests"] += 1
        return RateLimitResult(
            allowed=True,
            message="OK",
            remaining=max(0, remaining)
        )
    
    def _set_cooldown(self, key: str):
        """Set cooldown for a key."""
        if key:
            self._cooldowns[key] = time.time() + (self.config.cooldown_minutes * 60)
    
    def _auto_block_ip(self, ip: str):
        """Auto-block an IP due to abuse."""
        end_time = time.time() + (self.config.block_duration_minutes * 60)
        self._auto_blocked[ip] = end_time
        self._stats["blocked_ips_count"] = len(self._auto_blocked)
    
    # ==========================================================================
    # WHITELIST / BLACKLIST MANAGEMENT
    # ==========================================================================
    
    def add_to_whitelist(self, ip: str) -> bool:
        """Add IP to whitelist."""
        if not ip:
            return False
        self.config.whitelisted_ips.add(ip)
        # Remove from blacklist if present
        self.config.blacklisted_ips.discard(ip)
        return True
    
    def remove_from_whitelist(self, ip: str) -> bool:
        """Remove IP from whitelist."""
        if ip in self.config.whitelisted_ips:
            self.config.whitelisted_ips.discard(ip)
            return True
        return False
    
    def add_to_blacklist(self, ip: str) -> bool:
        """Add IP to blacklist."""
        if not ip:
            return False
        self.config.blacklisted_ips.add(ip)
        # Remove from whitelist if present
        self.config.whitelisted_ips.discard(ip)
        return True
    
    def remove_from_blacklist(self, ip: str) -> bool:
        """Remove IP from blacklist."""
        if ip in self.config.blacklisted_ips:
            self.config.blacklisted_ips.discard(ip)
            return True
        return False
    
    def unblock_ip(self, ip: str) -> bool:
        """Remove auto-block on IP."""
        if ip in self._auto_blocked:
            del self._auto_blocked[ip]
            self._stats["blocked_ips_count"] = len(self._auto_blocked)
            return True
        return False
    
    # ==========================================================================
    # PREMIUM USER MANAGEMENT
    # ==========================================================================
    
    def add_premium_user(self, user_id: str) -> bool:
        """Mark user as premium."""
        if user_id:
            self._premium_users.add(user_id)
            return True
        return False
    
    def remove_premium_user(self, user_id: str) -> bool:
        """Remove premium status."""
        if user_id in self._premium_users:
            self._premium_users.discard(user_id)
            return True
        return False
    
    def is_premium(self, user_id: str) -> bool:
        """Check if user is premium."""
        return user_id in self._premium_users
    
    # ==========================================================================
    # STATISTICS & MONITORING
    # ==========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            **self._stats,
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "requests_per_day": self.config.requests_per_day,
                "burst_limit": self.config.burst_limit,
            },
            "active_entries": {
                "ip_counters": len(self._ip_minute_counters),
                "user_counters": len(self._user_minute_counters),
                "cooldowns": len(self._cooldowns),
                "auto_blocked": len(self._auto_blocked),
            },
            "lists": {
                "whitelisted_count": len(self.config.whitelisted_ips),
                "blacklisted_count": len(self.config.blacklisted_ips),
                "premium_users_count": len(self._premium_users),
            }
        }
    
    def get_ip_status(self, ip: str) -> Dict[str, Any]:
        """Get status for specific IP."""
        now = time.time()
        
        is_blocked = ip in self.config.blacklisted_ips
        is_auto_blocked = ip in self._auto_blocked and self._auto_blocked[ip] > now
        is_whitelisted = ip in self.config.whitelisted_ips
        
        return {
            "ip": ip,
            "whitelisted": is_whitelisted,
            "blacklisted": is_blocked,
            "auto_blocked": is_auto_blocked,
            "auto_block_expires": (
                datetime.fromtimestamp(self._auto_blocked[ip]).isoformat()
                if is_auto_blocked else None
            ),
            "current_counts": {
                "minute": self._ip_minute_counters[ip].get_count() if ip in self._ip_minute_counters else 0,
                "hour": self._ip_hour_counters[ip].get_count() if ip in self._ip_hour_counters else 0,
                "day": self._ip_day_counters[ip].get_count() if ip in self._ip_day_counters else 0,
            }
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self._stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "blocked_requests": 0,
            "rate_limited_requests": 0,
            "blocked_ips_count": len(self._auto_blocked)
        }
    
    def reset_ip(self, ip: str):
        """Reset all counters for an IP."""
        for counter_dict in [
            self._ip_minute_counters,
            self._ip_hour_counters,
            self._ip_day_counters,
            self._burst_counters
        ]:
            if ip in counter_dict:
                counter_dict[ip].reset()
        
        # Clear cooldowns for this IP
        keys_to_remove = [k for k in self._cooldowns if ip in k]
        for k in keys_to_remove:
            del self._cooldowns[k]
    
    # ==========================================================================
    # CLEANUP
    # ==========================================================================
    
    def shutdown(self):
        """Shutdown rate limiter (stop cleanup thread)."""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)


# ==============================================================================
# SINGLETON ACCESSOR
# ==============================================================================

_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """
    Get Rate Limiter singleton instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(config)
    
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset singleton instance (for testing)."""
    global _rate_limiter
    if _rate_limiter:
        _rate_limiter.shutdown()
    _rate_limiter = None
    RateLimiter._instance = None


# ==============================================================================
# DECORATORS FOR FLASK/FASTAPI
# ==============================================================================

def rate_limit_decorator(
    limiter: Optional[RateLimiter] = None,
    get_ip_func=None,
    get_user_func=None
):
    """
    Decorator for Flask routes.
    
    Usage:
        @app.route('/api/chat')
        @rate_limit_decorator()
        def chat():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            rl = limiter or get_rate_limiter()
            
            # Get IP
            ip = None
            if get_ip_func:
                ip = get_ip_func()
            else:
                try:
                    from flask import request
                    ip = request.remote_addr
                except:
                    pass
            
            # Get user
            user_id = None
            if get_user_func:
                user_id = get_user_func()
            
            # Check rate limit
            result = rl.check_rate_limit(ip=ip, user_id=user_id)
            
            if not result.allowed:
                try:
                    from flask import jsonify
                    return jsonify(result.to_dict()), 429
                except:
                    return {"error": result.message}, 429
            
            return func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        return wrapper
    
    return decorator


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    print("\n Rate Limiter - Test Mode")
    print("=" * 50)
    
    # Test basic functionality
    limiter = RateLimiter(RateLimitConfig(
        requests_per_minute=5,  # Low for testing
        burst_limit=3
    ))
    
    test_ip = "192.168.1.100"
    
    print(f"\n Config: {limiter.config.requests_per_minute} req/min, burst={limiter.config.burst_limit}")
    print(f"\n Testing {test_ip}:")
    
    # Send multiple requests
    for i in range(10):
        result = limiter.check_rate_limit(ip=test_ip)
        status = "✅" if result.allowed else "❌"
        print(f"   Request {i+1}: {status} {result.message} (remaining: {result.remaining})")
        time.sleep(0.1)
    
    print(f"\n Statistics:")
    stats = limiter.get_statistics()
    print(f"   Total: {stats['total_requests']}")
    print(f"   Allowed: {stats['allowed_requests']}")
    print(f"   Rate limited: {stats['rate_limited_requests']}")
    
    print(f"\n IP Status:")
    ip_status = limiter.get_ip_status(test_ip)
    print(f"   Minute count: {ip_status['current_counts']['minute']}")
    
    # Cleanup
    limiter.shutdown()
    print("\n✅ Test completed!")

