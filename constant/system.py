#!/usr/bin/env python3

"""
System Configuration Constants

This module contains system-wide limits, timeouts, default configuration values,
and performance thresholds that define the operational boundaries and default
behaviors of the system.
"""
import ssl
from typing import Dict, Set

# Configuration files
DEFAULT_CONFIG_FILE: str = "config.yaml"
DEFAULT_WORKSPACE_DIR: str = "./data"

# Log levels
LOG_LEVEL_DEBUG: str = "DEBUG"
LOG_LEVEL_INFO: str = "INFO"
LOG_LEVEL_WARNING: str = "WARNING"
LOG_LEVEL_ERROR: str = "ERROR"

# Log cleanup configuration. True: delete logs, False: archive logs
DEFAULT_LOG_CLEANUP_DELETE: bool = True

# Application banner
APPLICATION_BANNER: str = """
╔══════════════════════════════════════════════════════════════╗
║                    Async Pipeline System                     ║
║              Multi-Provider API Key Discovery                ║
╚══════════════════════════════════════════════════════════════╝
"""

# Default intervals and timeouts
DEFAULT_STATS_INTERVAL: int = 15
DEFAULT_SHUTDOWN_TIMEOUT: float = 30.0

# Default configuration values
DEFAULT_BATCH_SIZE: int = 50
DEFAULT_SAVE_INTERVAL: int = 30
DEFAULT_QUEUE_INTERVAL: int = 60
DEFAULT_RETRIES: int = 3
DEFAULT_TIMEOUT: int = 30

# Search and API specific timeouts
GITHUB_API_TIMEOUT: int = 60  # Increased from 30 to 60
GITHUB_API_INTERVAL: int = 3  # Increased from 2 to 3
GITHUB_API_RATE_LIMIT_BACKOFF: int = 60  # 1 minute
GITHUB_WEB_COUNT_DELAY_MAX: float = 3.0  # Increased from 2.0 to 3.0
CHAT_RETRY_INTERVAL: int = 2  # Increased from 1 to 2
COLLECT_RETRY_INTERVAL: int = 2  # Increased from 1 to 2

# Signal handling timeouts
FORCE_EXIT_GRACE_PERIOD: float = 5.0  # Time to wait for second Ctrl+C

# Maximum number of times to re-queue when processing fails
DEFAULT_MAX_RETRIES_REQUEUED: int = 3

# Memory and performance thresholds
DEFAULT_MEMORY_THRESHOLD: int = 1024 * 1024 * 1024  # 1GB in bytes
DEFAULT_ERROR_RATE_THRESHOLD_APP: float = 0.1  # 10% error rate
DEFAULT_QUEUE_SIZE_THRESHOLD_APP: int = 1000  # Queue size threshold

# Alert configuration
ALERT_COOLDOWN_SECONDS: int = 300  # 5 minutes cooldown between alerts

# Load balancer configuration
DEFAULT_MIN_WORKERS: int = 1
DEFAULT_MAX_WORKERS: int = 10
DEFAULT_TARGET_QUEUE_SIZE: int = 100
DEFAULT_ADJUSTMENT_INTERVAL: float = 5.0  # seconds
DEFAULT_SCALE_UP_THRESHOLD: float = 0.8  # 80% queue utilization
DEFAULT_SCALE_DOWN_THRESHOLD: float = 0.2  # 20% queue utilization

# Performance monitoring
LB_RECENT_HISTORY_SIZE: int = 10  # Number of recent measurements to keep
PROGRESS_UPDATE_INTERVAL: float = 1.0  # Progress update interval in seconds

# SSL Context (secure by default)
CTX: ssl.SSLContext = ssl.create_default_context()
# Keep hostname checking and certificate verification enabled by default for security.
# If you need to disable verification for local testing, wire a config toggle at call sites.

# Default API paths
DEFAULT_COMPLETION_PATH: str = "/v1/chat/completions"
DEFAULT_MODEL_PATH: str = "/v1/models"
DEFAULT_AUTHORIZATION_HEADER: str = "Authorization"

# HTTP Configuration
DEFAULT_HEADERS: Dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}

# API Configuration
DEFAULT_QUESTION: str = "Hello"

# Error handling
NO_RETRY_ERROR_CODES: Set[int] = {400, 401, 402, 404, 422}


# Service types
SERVICE_TYPE_GITHUB_API: str = "github_api"
SERVICE_TYPE_GITHUB_WEB: str = "github_web"
PROVIDER_SERVICE_PREFIX: str = "provider"
