#!/usr/bin/env python3

"""
HTTP client utilities and GitHub-specific search functions for the search engine.
"""

import gzip
import itertools
import json
import random
import re
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

from core.models import Service

# --- Credential cooldown (simple in-process store) ---
_token_cooldown: Dict[str, float] = {}
_session_cooldown: Dict[str, float] = {}

def _now() -> float:
    return time.time()

def _cleanup_cooldown() -> None:
    now = _now()
    for store in (_token_cooldown, _session_cooldown):
        expired = [k for k, t in store.items() if t <= now]
        for k in expired:
            try:
                del store[k]
            except Exception:
                pass

def mark_token_cooldown(token: str, seconds: int = 900) -> None:
    if not token:
        return
    _cleanup_cooldown()
    _token_cooldown[token] = _now() + max(60, seconds)

def mark_session_cooldown(session: str, seconds: int = 900) -> None:
    if not session:
        return
    _cleanup_cooldown()
    _session_cooldown[session] = _now() + max(60, seconds)

def is_token_on_cooldown(token: Optional[str]) -> bool:
    if not token:
        return False
    _cleanup_cooldown()
    exp = _token_cooldown.get(token, 0)
    return exp > _now()

def is_session_on_cooldown(session: Optional[str]) -> bool:
    if not session:
        return False
    _cleanup_cooldown()
    exp = _session_cooldown.get(session, 0)
    return exp > _now()
from tools.logger import get_logger
from tools.utils import encoding_url, isblank, trim

logger = get_logger("search")

from constant.search import API_RESULTS_PER_PAGE, WEB_RESULTS_PER_PAGE
from constant.system import (
    CHAT_RETRY_INTERVAL,
    COLLECT_RETRY_INTERVAL,
    CTX,
    DEFAULT_HEADERS,
    DEFAULT_QUESTION,
    GITHUB_API_INTERVAL,
    GITHUB_API_RATE_LIMIT_BACKOFF,
    GITHUB_API_TIMEOUT,
    GITHUB_WEB_COUNT_DELAY_MAX,
    NO_RETRY_ERROR_CODES,
    SERVICE_TYPE_GITHUB_API,
    SERVICE_TYPE_GITHUB_WEB,
)
from core.exceptions import NetworkError, ValidationError
from core.models import RateLimitConfig
from core.types import IAuthProvider
from tools.coordinator import get_user_agent
from tools.ratelimit import RateLimiter
from tools.resources import managed_network
from tools.retry import network_retry
from tools.utils import handle_exceptions


class GitHubClient:
    """GitHub-specific HTTP client with rate limiting and dependency injection"""

    def __init__(self, limiter: Optional[RateLimiter] = None, resource_provider: Optional[IAuthProvider] = None):
        """Initialize GitHub client

        Args:
            limiter: Rate limiter for request throttling
            resource_provider: Resource provider for credentials and user agents
        """
        self.limiter = limiter
        self.resource_provider = resource_provider

    def _get_user_agent(self) -> str:
        """Get User-Agent string using dependency injection or fallback

        Returns:
            str: User-Agent string
        """
        if self.resource_provider:
            return self.resource_provider.get_user_agent()
        else:
            # Fallback to global function for backward compatibility
            return get_user_agent()

    def _service(self, url: str) -> Optional[str]:
        """Detect service type from URL"""
        if not url:
            return None

        url_lower = url.lower()
        if "api.github.com" in url_lower:
            return SERVICE_TYPE_GITHUB_API
        elif "github.com" in url_lower:
            return SERVICE_TYPE_GITHUB_WEB

        return None

    def _limit(self, service: str) -> bool:
        """Apply rate limiting, return True if request can proceed"""
        if not self.limiter or not service:
            return True

        # Try immediate acquisition
        if self.limiter.acquire(service):
            return True

        # Wait for tokens
        wait = self.limiter.wait_time(service)
        if wait > 0:
            bucket = self.limiter._get_bucket(service)
            max_value = bucket.burst if bucket else "unknown"
            logger.info(f"Rate limit hit for {service}, waiting {wait:.2f}s, max: {max_value}")
            time.sleep(wait)
            return self.limiter.acquire(service)

        return False

    def _report(self, service: str, success: bool) -> None:
        """Report request result for adaptive adjustment"""
        if self.limiter and service:
            self.limiter.report_result(service, success)

    def _handle_error(self, service: str, status: int, message: str) -> None:
        """Handle GitHub-specific errors"""
        if status == 403 and service == SERVICE_TYPE_GITHUB_API:
            if "rate limit" in message.lower():
                logger.info("GitHub API rate limit exceeded, backing off")
                time.sleep(GITHUB_API_RATE_LIMIT_BACKOFF)  # Wait for rate limit reset

    def get(
        self,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        retries: int = 3,
        interval: float = 0,
        timeout: float = 10,
    ) -> str:
        """Make rate-limited HTTP GET request to GitHub"""
        service = self._service(url)

        # Apply rate limiting
        if service and not self._limit(service):
            logger.debug(f"Rate limit acquisition failed for {service}")
            return ""

        # Jitter to avoid synchronized bursts
        try:
            time.sleep(random.uniform(0.08, 0.2))
        except Exception:
            pass

        # Make request using original http_get
        result = http_get(url, headers, params, retries, interval, timeout)
        success = bool(result)

        # Report result for adaptive adjustment
        self._report(service, success)

        return result


# Global GitHub client instance
_github_client: Optional[GitHubClient] = None


def init_github_client(limits: Dict[str, RateLimitConfig]) -> None:
    """Initialize GitHub client with rate limiter"""
    global _github_client
    limiter = RateLimiter(limits)
    _github_client = GitHubClient(limiter)
    logger.info("GitHub client initialized with rate limiting")


def get_github_client() -> GitHubClient:
    """Get GitHub client instance"""
    if not _github_client:
        # Fallback client without rate limiting
        return GitHubClient()
    return _github_client



def set_github_client_rate_limiter(limiter: RateLimiter) -> None:
    """Bind a shared RateLimiter instance to the GitHub client.

    This avoids creating a separate limiter and ensures unified rate control
    with the pipeline's global limiter.
    """
    global _github_client
    _github_client = GitHubClient(limiter)
    logger.info("GitHub client bound to shared rate limiter")


def get_github_stats() -> Dict[str, Dict[str, float]]:
    """Get rate limiter statistics"""
    if _github_client and _github_client.limiter:
        stats = _github_client.limiter.get_stats()
        # Convert back to dict format for backward compatibility
        result = {}
        for service, bucket_stats in stats.services.items():
            result[service] = {
                "rate": bucket_stats.rate,
                "burst": bucket_stats.burst,
                "tokens": bucket_stats.tokens,
                "utilization": bucket_stats.utilization,
                "consecutive_success": bucket_stats.consecutive_success,
                "consecutive_failures": bucket_stats.consecutive_failures,
                "adaptive": bucket_stats.adaptive,
                "original_rate": bucket_stats.original_rate,
            }
        return result
    return {}


def log_github_stats() -> None:
    """Log current rate limiter statistics"""
    if not _github_client or not _github_client.limiter:
        return

    stats = _github_client.limiter.get_stats()
    for service, bucket_stats in stats.services.items():
        logger.info(
            f"Rate limiter [{service}]: rate={bucket_stats.rate:.2f}/s, "
            f"tokens={bucket_stats.tokens:.1f}/{bucket_stats.burst}, "
            f"utilization={bucket_stats.utilization:.1%}"
        )


@network_retry
def http_get(
    url: str,
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    retries: int = 5,
    interval: float = 1.0,
    timeout: float = 60,
) -> str:
    """HTTP GET request with configurable retry handling

    Args:
        url: URL to request
        headers: HTTP headers
        params: URL parameters
        retries: Number of retry attempts (default: 3, minimum: 1)
        interval: Initial delay between retries in seconds (default: 1.0, minimum: 0.1)
        timeout: Request timeout in seconds

    Returns:
        str: Response content

    Raises:
        ValidationError: For invalid input
        NetworkError: For network-related issues
        ConnectionError: For connection failures (will be retried)
        TimeoutError: For timeout errors (will be retried)

    Note:
        The @network_retry decorator automatically extracts retries and interval
        parameters to configure retry behavior dynamically. Retry logic uses
        exponential backoff with jitter for optimal performance.
    """
    # Input validation
    if isblank(url):
        raise ValidationError("URL cannot be empty", field="url")

    # Setup request
    headers = headers or DEFAULT_HEADERS.copy()
    timeout = max(1, timeout)

    try:
        # Encode URL and add parameters
        encoded_url = encoding_url(url)
        if params and isinstance(params, dict):
            data = urllib.parse.urlencode(params)
            separator = "&" if "?" in encoded_url else "?"
            encoded_url += f"{separator}{data}"

        # Make request with managed network resource
        request = urllib.request.Request(url=encoded_url, headers=headers)
        with managed_network(
            urllib.request.urlopen(request, timeout=timeout, context=CTX), "http_connection"
        ) as response:
            # Handle response
            content = response.read()
            status_code = response.getcode()

            if status_code != 200:
                raise NetworkError(f"HTTP {status_code} error for URL: {url}")

            # Decode content
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    return gzip.decompress(content).decode("utf-8")
                except Exception:
                    raise NetworkError("Failed to decode response content")

    except urllib.error.HTTPError as e:
        # Handle HTTP errors with classification and body inspection for GitHub abuse
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore") if hasattr(e, "read") else ""
        except Exception:
            body = ""
        lower = (body or e.reason or "").lower()
        if e.code == 429 or (e.code == 403 and ("abuse" in lower or "secondary rate limit" in lower)):
            # Rate limit / abuse detection should be retried
            raise ConnectionError(f"Retryable limit (HTTP {e.code}): {e.reason or body}")
        elif e.code in (401, 403):
            # Auth errors should not be retried; additionally mark credential cooldown
            cred = headers.get("Authorization") if headers else None
            if cred and cred.lower().startswith("token "):
                mark_token_cooldown(cred.split(" ",1)[1])
            # For web (no Authorization), upper stack will call mark_session_cooldown when seeing login page or 401
            raise NetworkError(f"Authentication failed (HTTP {e.code})")
        elif e.code >= 500:
            # Server errors should be retried
            raise ConnectionError(f"Server error (HTTP {e.code}): {e.reason}")
        else:
            # Client errors should not be retried
            raise NetworkError(f"HTTP {e.code} error: {e.reason}")

    except urllib.error.URLError as e:
        # URL errors are usually network-related and should be retried
        raise ConnectionError(f"URL error: {e.reason}")

    except Exception as e:
        # Classify unknown errors
        if "timeout" in str(e).lower():
            # Timeout errors should be retried
            raise TimeoutError(f"Request timeout: {e}")
        else:
            # Other errors should not be retried
            raise NetworkError(f"Unexpected error: {e}")


def chat(
    url: str, headers: Dict, model: str = "", params: Optional[Dict] = None, retries: int = 2, timeout: int = 10
) -> Tuple[int, str]:
    """Make chat API request with retry logic."""

    def output(code: int, message: str, debug: bool = False) -> None:
        text = f"[chat] failed to request URL: {url}, headers: {headers}, status code: {code}, message: {message}"
        if debug:
            logger.debug(text)
        else:
            logger.error(text)

    url, model = trim(url), trim(model)
    if not url:
        logger.error("[chat] url cannot be empty")
        return 400, None

    if not isinstance(headers, dict):
        logger.error("[chat] headers must be a dict")
        return 400, None
    elif len(headers) == 0:
        headers["content-type"] = "application/json"

    if not params or not isinstance(params, dict):
        if not model:
            logger.error("[chat] model cannot be empty")
            return 400, None

        params = {
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": DEFAULT_QUESTION}],
        }

    payload = json.dumps(params).encode("utf8")
    timeout = max(1, timeout)
    retries = max(1, retries)
    code, message, attempt = 400, None, 0

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    while attempt < retries:
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as response:
                code = 200
                message = response.read().decode("utf8")
                break
        except urllib.error.HTTPError as e:
            code = e.code
            if code != 401:
                try:
                    # read response body
                    message = e.read().decode("utf8")

                    # not a json string, use reason instead
                    if not message.startswith("{") or not message.endswith("}"):
                        message = e.reason
                except Exception:
                    message = e.reason

                # print http status code and error message
                output(code=code, message=message, debug=False)

            if code in NO_RETRY_ERROR_CODES:
                break
        except Exception:
            output(code=code, message=traceback.format_exc(), debug=True)

        attempt += 1
        time.sleep(CHAT_RETRY_INTERVAL)

    return code, message


def search_github_web(query: str, session: str, page: int) -> str:
    """Use github web search instead of rest api due to it not support regex syntax."""
    if page <= 0 or isblank(session) or isblank(query):
        return ""

    url = f"https://github.com/search?o=desc&p={page}&type=code&q={query}"
    headers: Dict[str, str] = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Referer": "https://github.com",
        "User-Agent": get_user_agent(),
        "Cookie": f"user_session={session}",
    }

    # Skip sessions that are currently on cooldown
    if is_session_on_cooldown(session):
        return ""

    client = get_github_client()
    content = client.get(url=url, headers=headers)
    if re.search(r"<h1>Sign in to GitHub</h1>", content, flags=re.I):
        logger.error("[GithubCrawl] Session has expired, please provide a valid session and try again")
        mark_session_cooldown(session, seconds=1800)
        return ""

    if re.search(r"(abuse|secondary rate limit|too many requests)", content or "", flags=re.I):
        mark_session_cooldown(session, seconds=900)

    return content


def search_github_api(query: str, token: str, page: int = 1, peer_page: int = API_RESULTS_PER_PAGE) -> List[str]:
    """Rate limit: 10RPM."""
    if isblank(token) or isblank(query):
        return []

    peer_page, page = min(max(peer_page, 1), API_RESULTS_PER_PAGE), max(1, page)
    url = f"https://api.github.com/search/code?q={query}&sort=indexed&order=desc&per_page={peer_page}&page={page}"
    headers: Dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # jitter to avoid secondary rate limit bursts
    try:
        time.sleep(random.uniform(0.2, 0.6))
    except Exception:
        pass

    # Skip credentials that are currently on cooldown
    if is_token_on_cooldown(token):
        return []

    client = get_github_client()
    content = client.get(url=url, headers=headers, interval=GITHUB_API_INTERVAL, timeout=GITHUB_API_TIMEOUT)
    if isblank(content):
        return []
    try:
        items = json.loads(content).get("items", [])
        links: set[str] = set()

        for item in items:
            if not item or type(item) != dict:
                continue

            link = item.get("html_url", "")
            if isblank(link):
                continue
            links.add(link)

        return list(links)
    except Exception:
        return []


def search_web_with_count(
    query: str,
    session: str,
    page: int = 1,
    callback: Optional[Callable[[List[str], str], None]] = None,
) -> Tuple[List[str], int, str]:
    """
    Search GitHub web and return results, total count, and content.
    Returns: (results_list, total_count, content)
    """
    if page <= 0 or isblank(session) or isblank(query):
        return [], 0, ""

    # Get results from web search
    content = search_github_web(query, session, page)
    if isblank(content):
        return [], 0, ""

    # Extract links from content
    try:
        # Capture blob links with or without line anchors
        regex = r'href="(/[^\s\"]+/blob/[^\"#]+)(?:#L\d+)?"'
        groups = re.findall(regex, content, flags=re.I)
        uris = list(set(groups)) if groups else []
        links = set()

        for uri in uris:
            links.add(f"https://github.com{uri}")

        results = list(links)
    except Exception:
        results = []

    # Call extract callback if provided
    if callback and isinstance(callback, Callable) and results:
        try:
            callback(results, content)
        except Exception as e:
            logger.error(f"[search] callback failed: {e}")

    # Get total count (only for first page to avoid redundant calls)
    if page == 1:
        total = estimate_web_total(query, session, content)
    else:
        # For non-first pages, we don't need total count, use 0 as placeholder
        total = 0

    return results, total, content


def search_api_with_count(
    query: str, token: str, page: int = 1, peer_page: int = API_RESULTS_PER_PAGE
) -> Tuple[List[str], int, str]:
    """
    Search GitHub API and return results, total count, and raw content.

    Args:
        query: Search query string
        token: GitHub API token for authentication
        page: Page number to retrieve (default: 1)
        peer_page: Results per page (default: API_RESULTS_PER_PAGE)

    Returns:
        Tuple containing:
        - List[str]: List of GitHub URLs found
        - int: Total count of results available
        - str: Raw JSON response content
    """
    if isblank(token) or isblank(query):
        return [], 0, ""

    peer_page, page = min(max(peer_page, 1), API_RESULTS_PER_PAGE), max(1, page)
    url = f"https://api.github.com/search/code?q={query}&sort=indexed&order=desc&per_page={peer_page}&page={page}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    client = get_github_client()
    content = client.get(url=url, headers=headers, interval=GITHUB_API_INTERVAL, timeout=GITHUB_API_TIMEOUT)
    if isblank(content):
        return [], 0, ""

    try:
        data = json.loads(content)
        items = data.get("items", [])
        total = data.get("total_count", 0)

        links = set()
        for item in items:
            if not item or type(item) != dict:
                continue

            link = item.get("html_url", "")
            if isblank(link):
                continue
            links.add(link)

        return list(links), total, content
    except Exception:
        return [], 0, ""


def search_with_count(
    query: str,
    session: str,
    page: int,
    with_api: bool,
    peer_page: int,
    callback: Optional[Callable[[List[str], str], None]] = None,
) -> Tuple[List[str], int, str]:
    """
    Unified search interface that returns results, total count, and content.
    Returns: (results_list, total_count, content)
    """
    keywords = urllib.parse.quote_plus(query)
    if with_api:
        return search_api_with_count(keywords, session, page, peer_page)
    else:
        return search_web_with_count(keywords, session, page, callback)


@handle_exceptions(default_result=0, log_level="error")
def get_total_num(query: str, token: str) -> int:
    """Get total number of results from GitHub API."""
    if isblank(token) or isblank(query):
        return 0

    url = f"https://api.github.com/search/code?q={query}&sort=indexed&order=desc&per_page=20&page=1"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    client = get_github_client()
    content = client.get(url=url, headers=headers, interval=1)
    data = json.loads(content)
    return data.get("total_count", 0)


def estimate_web_total(query: str, session: str, content: Optional[str] = None) -> int:
    """
    Get total count for web search using GitHub's blackbird_count API.
    Performs a single search and then queries the count API.
    """
    if isblank(session) or isblank(query):
        return 0

    try:
        message = urllib.parse.unquote_plus(query)
    except Exception:
        message = query

    try:
        if content is None:
            # Perform initial search to trigger count calculation and get content for fallback
            content = search_github_web(query=query, session=session, page=1)

        content = trim(content)
        if not content:
            logger.warning(f"[search] initial search failed for query: {message}, using conservative estimate")
            # Conservative estimate
            return WEB_RESULTS_PER_PAGE

        # Check if query is already encoded to avoid double encoding
        if "%" in query and any(c in query for c in ["%2F", "%5B", "%5D", "%7B", "%7D"]):
            encoded = query.replace(" ", "+")
        else:
            encoded = urllib.parse.quote_plus(query)

        # Query the blackbird_count API
        url = f"https://github.com/search/blackbird_count?saved_searches=^&q={encoded}"
        headers = {
            "User-Agent": get_user_agent(),
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": f"https://github.com/search?q={encoded}^&type=code",
            "X-Requested-With": "XMLHttpRequest",
            "Cookie": f"user_session={session}",
        }

        # Random delay to ensure count is calculated
        time.sleep(random.random() * GITHUB_WEB_COUNT_DELAY_MAX)

        client = get_github_client()
        response = client.get(url=url, headers=headers, interval=1)
        if response:
            data = json.loads(response)
            if not data.get("failed", True):
                raw = data.get("count", 0)
                try:
                    count = int(str(raw).replace(",", "").strip())
                except Exception:
                    count = 0
                mode = data.get("mode", "unknown")
                logger.info(f"[search] got {count} results, mode: {mode}, query: {message}")

                # Return count if valid, otherwise try page extraction
                return count if count > 0 else extract_count_from_page(content, query)

        # Fallback: extract count from search page
        return extract_count_from_page(content, query)

    except Exception as e:
        # Downgrade log level for expected auth/limit issues during web count (handled by cooldown elsewhere)
        msg = str(e)
        if any(x in msg for x in ["HTTP 401", "HTTP 403", "HTTP 429", "abuse", "secondary rate limit"]):
            logger.warning(f"[search] estimation fallback for query: {message}, reason: {e}, using conservative estimate")
        else:
            logger.error(f"[search] estimation failed for query: {message}, error: {e}, using conservative estimate")
        # Conservative estimate
        return WEB_RESULTS_PER_PAGE


def extract_count_from_page(content: str, query: str) -> int:
    """Extract result count from GitHub search page content."""
    if isblank(content):
        return WEB_RESULTS_PER_PAGE

    try:
        message = urllib.parse.unquote_plus(query)

        # Try different patterns GitHub uses to show result counts
        patterns = [
            r"We\'ve found ([\d,]+) code results",
            r"([\d,]+) code results",
            r'data-total-count="([\d,]+)"',
            r'"total_count":(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.I)
            if match:
                text = match.group(1).replace(",", "")
                count = int(text)
                logger.info(f"[search] extracted {count} results from page for query: {message}")
                return count

        # If no count found, use conservative estimate
        logger.warning(f"[search] could not extract count from page for query: {message}")
        return WEB_RESULTS_PER_PAGE

    except Exception as e:
        logger.error(f"[search] failed to extract count from page: {e}")
        return WEB_RESULTS_PER_PAGE


def search_code(
    query: str,
    session: str,
    page: int,
    with_api: bool,
    peer_page: int,
    callback: Optional[Callable[[List[str], str], None]] = None,
) -> Tuple[List[str], str]:
    """
    Search code with unified interface.
    Returns: (results_list, content)
    """
    keyword = urllib.parse.quote_plus(trim(query))
    if not keyword:
        return [], ""

    if with_api:
        results = search_github_api(query=keyword, token=session, page=page, peer_page=peer_page)
        return results, ""  # API doesn't provide page content

    content = search_github_web(query=keyword, session=session, page=page)
    if isblank(content):
        return [], ""

    try:
        # Capture blob links with or without line anchors
        regex = r'href="(/[^\s\"]+/blob/[^\"#]+)(?:#L\d+)?"'
        groups = re.findall(regex, content, flags=re.I)
        uris = list(set(groups)) if groups else []
        links = set()

        for uri in uris:
            links.add(f"https://github.com{uri}")

        results = list(links)

        # Call extract callback if provided
        if callback and isinstance(callback, Callable) and results:
            try:
                callback(results, content)
            except Exception as e:
                logger.error(f"[search] callback failed: {e}")

        return results, content
    except Exception:
        return [], ""


@handle_exceptions(default_result=[], log_level="error")
def collect(
    key_pattern: str,
    url: str = "",
    retries: int = 3,
    address_pattern: str = "",
    endpoint_pattern: str = "",
    model_pattern: str = "",
    text: Optional[str] = None,
) -> List[Service]:
    """Extract API keys and related information from URLs or text content

    Args:
        key_pattern: Regex pattern to match API keys
        url: URL to fetch content from (if text not provided)
        retries: Number of retry attempts for HTTP requests
        address_pattern: Regex pattern to match service addresses
        endpoint_pattern: Regex pattern to match endpoints
        model_pattern: Regex pattern to match model names
        text: Text content to search (if provided, url is ignored)

    Returns:
        List[Service]: List of Service objects with extracted information
    """
    if (not isinstance(url, str) and not isinstance(text, str)) or not isinstance(key_pattern, str):
        return []

    # Path-level filtering for noisy locations
    def _is_noisy_path(u: str) -> bool:
        try:
            if not isinstance(u, str):
                return False
            low = u.lower()
            noisy = ("example", "examples", "sample", "samples", "mock", "mocks", "test", "tests", "fixture", "fixtures", "spec", "specs", "docs", "tutorial")
            return any(x in low for x in noisy)
        except Exception:
            return False

    # Track GitHub context for repo-aware pairing
    _repo_owner = _repo_name = _repo_branch = _repo_file = ""

    if text:
        content = text
    else:
        # Prefer fetching plain/raw content from GitHub code pages to improve regex hit rate
        fetch_url = url
        try:
            u = trim(url)
            if _is_noisy_path(u):
                # 不立即丢弃：若后续形成配对再判定，这里只降低优先级
                pass
            if u.startswith("https://github.com/") and "/blob/" in u:
                # Strip line anchors and force plain text rendering
                base = u.split("#", 1)[0]
                fetch_url = base + ("&plain=1" if "?" in base and "plain=1" not in base else ("?plain=1" if "?" not in base else base))
                try:
                    parts = urllib.parse.urlparse(u)
                    segs = [s for s in parts.path.split("/") if s]
                    if len(segs) >= 5 and segs[2] == "blob":
                        _repo_owner, _repo_name, _repo_branch = segs[0], segs[1], segs[3]
                        _repo_file = "/".join(segs[4:])
                        # 同理：不在此处直接丢弃，后续配对时再过滤
                        pass
                except Exception:
                    pass
        except Exception:
            fetch_url = url

        try:
            content = http_get(url=fetch_url, retries=retries, interval=COLLECT_RETRY_INTERVAL)
        except Exception:
            # Treat network errors as empty content so we can try fallback
            content = ""

        # Fallback: try raw.githubusercontent.com if plain fetch failed
        if not content and url.startswith("https://github.com/") and "/blob/" in url:
            try:
                parts = urllib.parse.urlparse(url)
                segs = [s for s in parts.path.split("/") if s]
                # Expect: /{owner}/{repo}/blob/{branch}/...path
                if len(segs) >= 5 and segs[2] == "blob":
                    _repo_owner, _repo_name, _repo_branch = segs[0], segs[1], segs[3]
                    _repo_file = "/".join(segs[4:])
                    if _is_noisy_path(_repo_file):
                        return []
                    raw_url = f"https://raw.githubusercontent.com/{_repo_owner}/{_repo_name}/{_repo_branch}/{_repo_file}"
                    try:
                        content = http_get(url=raw_url, retries=retries, interval=COLLECT_RETRY_INTERVAL)
                    except Exception:
                        content = ""
            except Exception:
                pass

    if not content:
        return []

    # Precompute meta for repo-aware pairing/enrichment
    _meta_base = {}
    try:
        if _repo_owner and _repo_name:
            _meta_base = {
                "repo_owner": _repo_owner,
                "repo_name": _repo_name,
                "branch": _repo_branch,
                "file_path": _repo_file,
                "source_url": url or "",
            }
    except Exception:
        _meta_base = {}

    # Helper: compute line number for a given offset
    def _line_no_at(pos: int) -> int:
        return content.count("\n", 0, max(0, pos)) + 1

    # Helper: check placeholder words in a text window
    _placeholder_re = re.compile(r"\b(example|sample|dummy|placeholder|your|xxxx+|test|mock|fake|invalid|notreal|redacted|todo|fixme)\b", re.I)

    def _has_placeholder(around: str) -> bool:
        return bool(_placeholder_re.search(around))

    # extract keys from content (with positions)
    key_pattern = trim(key_pattern)
    if not key_pattern:
        return []

    # When endpoint_pattern is provided, do near-proximity pairing to avoid cross-file mismatches
    endpoint_pattern = trim(endpoint_pattern)

    candidates: List[Service] = []

    if endpoint_pattern:
        try:
            key_re = re.compile(key_pattern, flags=re.I | re.S)
        except Exception:
            key_re = re.compile(key_pattern)
        try:
            ep_re = re.compile(endpoint_pattern, flags=re.I | re.S)
        except Exception:
            ep_re = re.compile(endpoint_pattern)

        # Simple placeholder filter to drop obvious fake/template keys
        def _is_placeholder(s: str) -> bool:
            try:
                t = str(s or "").strip().lower()
                if not t:
                    return True
                bad_sub = ("your", "example", "sample", "test", "mock", "dummy", "xxxx", "xxxxx", "replace")
                if any(x in t for x in bad_sub):
                    return True
                # common forms like sk-ant-your-...-here or api_key_here
                if re.findall(r"(your.*key|key.*here|api[_-]?key.*here|xxx+|^sk-[-_x]+$)", t, flags=re.I):
                    return True
                return False
            except Exception:
                return False

        key_matches: List[Tuple[str, int, int]] = []  # (value, start, end)
        for m in key_re.finditer(content):
            val = m.group(1) if m.groups() else m.group(0)
            if val:
                v = trim(val)
                if not _is_placeholder(v):
                    key_matches.append((v, m.start(), m.end()))

        ep_matches: List[Tuple[str, int, int]] = []
        for m in ep_re.finditer(content):
            val = m.group(1) if m.groups() else m.group(0)
            if val:
                v = trim(val)
                if not _is_placeholder(v):
                    ep_matches.append((v, m.start(), m.end()))

        if not key_matches or not ep_matches:
            # Emit half-candidates for repo-aware cross-file pairing
            half_candidates: List[Service] = []
            # Optional address/model extraction to carry along
            address_pattern = trim(address_pattern)
            addresses = extract(text=content, regex=address_pattern) if address_pattern else [""]
            if not addresses:
                addresses = [""]
            model_pattern = trim(model_pattern)
            models = extract(text=content, regex=model_pattern) if model_pattern else [""]
            if not models:
                models = [""]

            if key_matches and not ep_matches:
                # key-only
                for k_val, _, _ in key_matches:
                    for address, model in itertools.product(addresses, models):
                        svc = Service(address=address, endpoint="", key=k_val, model=model)
                        try:
                            if _meta_base:
                                svc.meta.update(_meta_base)
                            svc.meta["pair_half"] = "key_only"
                        except Exception:
                            pass
                        half_candidates.append(svc)
            elif ep_matches and not key_matches:
                # endpoint-only
                for e_val, _, _ in ep_matches:
                    for address, model in itertools.product(addresses, models):
                        svc = Service(address=address, endpoint=e_val, key="", model=model)
                        try:
                            if _meta_base:
                                svc.meta.update(_meta_base)
                            svc.meta["pair_half"] = "endpoint_only"
                        except Exception:
                            pass
                        half_candidates.append(svc)
            return half_candidates

        # Optional address/model extraction (global in file)
        address_pattern = trim(address_pattern)
        addresses = extract(text=content, regex=address_pattern) if address_pattern else [""]
        if address_pattern and not addresses:
            return []
        if not addresses:
            addresses = [""]

        model_pattern = trim(model_pattern)
        models = extract(text=content, regex=model_pattern) if model_pattern else [""]
        if model_pattern and not models:
            return []
        if not models:
            models = [""]

        # Dynamic neighbor threshold: tighten when context is noisy/long lines
        MAX_LINE_DISTANCE = 10
        try:
            # crude heuristic: if file seems long and has many comment markers, tighten
            lines = content.splitlines()
            n = len(lines)
            comment_like = sum(1 for ln in lines if ln.strip().startswith(('#','//','/*','*')))
            ratio = (comment_like / max(1, n))
            if n > 800 or ratio > 0.25:
                MAX_LINE_DISTANCE = 6
        except Exception:
            pass
        CONTEXT_LINES = 5

        # Build near-proximity pairs only (multi-tier threshold); widen progressively if过严
        matched_any = False
        # tier 1: tight
        for k_val, k_s, k_e in key_matches:
            k_ln = _line_no_at(k_s)
            for e_val, e_s, e_e in ep_matches:
                e_ln = _line_no_at(e_s)
                if abs(k_ln - e_ln) <= MAX_LINE_DISTANCE:
                    start = max(0, content.rfind("\n", 0, min(k_s, e_s)))
                    end = content.find("\n", max(k_e, e_e))
                    if end == -1:
                        end = len(content)
                    ctx = content[max(0, start):min(len(content), end)]
                    if _has_placeholder(ctx):
                        continue
                    for address, model in itertools.product(addresses, models):
                        svc = Service(address=address, endpoint=e_val, key=k_val, model=model)
                        try:
                            if _meta_base:
                                svc.meta.update(_meta_base)
                        except Exception:
                            pass
                        candidates.append(svc)
                        matched_any = True
        if matched_any:
            return candidates

        # tier 2: medium (double the distance)
        MED_DIST = min(12, MAX_LINE_DISTANCE * 2)
        for k_val, k_s, k_e in key_matches:
            k_ln = _line_no_at(k_s)
            for e_val, e_s, e_e in ep_matches:
                e_ln = _line_no_at(e_s)
                if abs(k_ln - e_ln) <= MED_DIST:
                    start = max(0, content.rfind("\n", 0, min(k_s, e_s)))
                    end = content.find("\n", max(k_e, e_e))
                    if end == -1:
                        end = len(content)
                    ctx = content[max(0, start):min(len(content), end)]
                    if _has_placeholder(ctx):
                        continue
                    for address, model in itertools.product(addresses, models):
                        svc = Service(address=address, endpoint=e_val, key=k_val, model=model)
                        try:
                            if _meta_base:
                                svc.meta.update(_meta_base)
                        except Exception:
                            pass
                        candidates.append(svc)
                        matched_any = True
        if matched_any:
            return candidates

        # tier 3: same file fallback (still require both present but without line distance)
        for k_val, _, _ in key_matches:
            for e_val, _, _ in ep_matches:
                for address, model in itertools.product(addresses, models):
                    svc = Service(address=address, endpoint=e_val, key=k_val, model=model)
                    try:
                        if _meta_base:
                            svc.meta.update(_meta_base)
                    except Exception:
                        pass
                    candidates.append(svc)
        return candidates

    # Fallback: original wide extraction without near pairing
    keys = extract(text=content, regex=key_pattern)
    if not keys:
        return []

    address_pattern = trim(address_pattern)
    addresses = extract(text=content, regex=address_pattern)
    if address_pattern and not addresses:
        return []
    if not addresses:
        addresses.append("")

    model_pattern = trim(model_pattern)
    models = extract(text=content, regex=model_pattern)
    need_model = False
    if model_pattern and not models:
        # Do not drop; allow half-candidate (e.g., Iflytek has key but missing APPID)
        need_model = True
        models = [""]
    if not models:
        models.append("")

    # endpoints (optional when not provided)
    endpoints: List[str] = []
    if endpoint_pattern:
        endpoints = extract(text=content, regex=endpoint_pattern)
        if endpoint_pattern and not endpoints:
            return []
    if not endpoints:
        endpoints.append("")

    for key, address, endpoint, model in itertools.product(keys, addresses, endpoints, models):
        candidates.append(Service(address=address, endpoint=endpoint, key=key, model=model))

    return candidates


@handle_exceptions(default_result=[], log_level="error")
def extract(text: str, regex: str) -> List[str]:
    """Extract strings from text using regex pattern."""
    # Defensive type checks to avoid regex errors
    if not isinstance(text, str) or not isinstance(regex, str):
        return []

    content, pattern = trim(text), trim(regex)
    if not content or not pattern:
        return []

    items: set[str] = set()
    # Use case-insensitive and DOTALL to catch keys split by whitespace/newlines
    try:
        groups = re.findall(pattern, content, flags=re.I | re.S)
    except TypeError:
        # Pattern may already include flags; fallback to default
        try:
            groups = re.findall(pattern, content)
        except Exception:
            return []
    for x in groups:
        words: List[str] = []
        if isinstance(x, str):
            words.append(x)
        elif isinstance(x, (tuple, list)):
            words.extend(list(x))
        else:
            logger.error(f"Unknown type: {type(x)}, value: {x}. Please optimize your regex")
            continue

        for word in words:
            key = trim(word)
            if key:
                items.add(key)

    return list(items)
