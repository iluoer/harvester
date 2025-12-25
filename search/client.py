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
    retries: int = 3,
    interval: float = 1.0,
    timeout: float = 10,
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
        # Handle HTTP errors with basic classification
        if e.code == 429:
            # Rate limit errors should be retried
            raise ConnectionError(f"Rate limit exceeded (HTTP {e.code})")
        elif e.code in (401, 403):
            # Auth errors should not be retried
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
                    message = e.read().decode("utf8").removeprefix("\n").removesuffix("\n")

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

    client = get_github_client()
    content = client.get(url=url, headers=headers)
    if re.search(r"<h1>Sign in to GitHub</h1>", content, flags=re.I):
        logger.error("[GithubCrawl] Session has expired, please provide a valid session and try again")
        return ""

    return content


def search_github_api(query: str, token: str, page: int = 1, peer_page: int = API_RESULTS_PER_PAGE) -> List[str]:
    """Rate limit: 10RPM."""
    if isblank(token) or isblank(query):
        return []

    peer_page, page = min(max(peer_page, 1), API_RESULTS_PER_PAGE), max(1, page)
    url = f"https://api.github.com/search/code?q={query}&sort=indexed&order=desc&per_page={peer_page}&page={page}"
    headers: Dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

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
        regex = r'href="(/[^\s"]+/blob/(?:[^"]+)?)#L\d+"'
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
        "Authorization": f"Bearer {token}",
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
        "Authorization": f"Bearer {token}",
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
                count = data.get("count", 0)
                mode = data.get("mode", "unknown")
                logger.info(f"[search] got {count} results, mode: {mode}, query: {message}")

                # Return count if valid, otherwise try page extraction
                return count if count > 0 else extract_count_from_page(content, query)

        # Fallback: extract count from search page
        return extract_count_from_page(content, query)

    except Exception as e:
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
        regex = r'href="(/[^\s"]+/blob/(?:[^"]+)?)#L\d+"'
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

    if text:
        content = text
    else:
        content = http_get(url=url, retries=retries, interval=COLLECT_RETRY_INTERVAL)

    if not content:
        return []

    # extract keys from content
    key_pattern = trim(key_pattern)
    keys = extract(text=content, regex=key_pattern)
    if not keys:
        return []

    # extract api addresses from content
    address_pattern = trim(address_pattern)
    addresses = extract(text=content, regex=address_pattern)
    if address_pattern and not addresses:
        return []
    if not addresses:
        addresses.append("")

    # extract api endpoints from content
    endpoint_pattern = trim(endpoint_pattern)
    endpoints = extract(text=content, regex=endpoint_pattern)
    if endpoint_pattern and not endpoints:
        return []
    if not endpoints:
        endpoints.append("")

    # extract models from content
    model_pattern = trim(model_pattern)
    models = extract(text=content, regex=model_pattern)
    if model_pattern and not models:
        return []
    if not models:
        models.append("")

    candidates = list()

    # combine keys, addresses and endpoints
    for key, address, endpoint, model in itertools.product(keys, addresses, endpoints, models):
        candidates.append(Service(address=address, endpoint=endpoint, key=key, model=model))

    return candidates


@handle_exceptions(default_result=[], log_level="error")
def extract(text: str, regex: str) -> List[str]:
    """Extract strings from text using regex pattern."""
    content, pattern = trim(text), trim(regex)
    if not content or not pattern:
        return []

    items: set[str] = set()
    groups = re.findall(pattern, content)
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
