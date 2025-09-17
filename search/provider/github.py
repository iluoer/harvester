#!/usr/bin/env python3

"""
GitHub provider for validating Personal Access Tokens (PAT).

Search stage finds potential github_pat_... tokens; check stage calls GitHub API /user
using the token to verify validity; inspect stage can list accessible repositories.
"""

import json
import time
import urllib.request
from urllib.error import HTTPError, URLError
from typing import Dict, List, Optional

from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.coordinator import get_user_agent
from tools.logger import get_logger
from tools.utils import handle_exceptions, trim

from .base import AIBaseProvider
from .registry import register_provider

logger = get_logger("provider")


class GitHubProvider(AIBaseProvider):
    """GitHub provider for validating Personal Access Tokens."""

    def __init__(self, conditions: List[Condition], **kwargs):
        # Consume possible config-provided values to avoid duplicate kwargs to base __init__
        name = trim(kwargs.pop("name", "github_pat"))
        base_url = trim(kwargs.pop("base_url", "")) or "https://api.github.com"
        completion_path = trim(kwargs.pop("completion_path", "")) or "/user"       # validate token
        model_path = trim(kwargs.pop("model_path", "")) or "/user/repos"            # list repos for inspection
        # AIBaseProvider requires non-empty default_model even if unused here
        default_model = trim(kwargs.pop("default_model", "")) or "default"

        super().__init__(
            name,
            base_url,
            completion_path,
            model_path,
            default_model,
            conditions,
            **kwargs,
        )

    def _headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        # Backward-compat helper; prefer _get_headers across codebase
        return self._get_headers(token, additional)

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        token = trim(token)
        if not token:
            return None
        if not isinstance(additional, dict):
            additional = {}
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {token}",
            "User-Agent": get_user_agent(),
        }
        headers.update(additional)
        return headers

    def _judge(self, code: int, message: str) -> CheckResult:
        msg = trim(message)
        if code == 200:
            try:
                data = json.loads(msg)
                if isinstance(data, dict) and data.get("login"):
                    return CheckResult.success(f"Valid GitHub PAT for user: {data.get('login')}")
            except Exception:
                logger.error(f"Failed to parse GitHub API response: {msg}")
                return CheckResult.fail(ErrorReason.UNKNOWN)
        elif code == 401:
            if "bad credentials" in msg.lower():
                return CheckResult.fail(ErrorReason.INVALID_KEY, "Invalid GitHub PAT")
            return CheckResult.fail(ErrorReason.INVALID_KEY)
        elif code == 403:
            low = msg.lower()
            if "rate limit" in low:
                return CheckResult.fail(ErrorReason.RATE_LIMITED, "GitHub API rate limit exceeded")
            if "expired" in low or "revoked" in low:
                return CheckResult.fail(ErrorReason.INVALID_KEY, "GitHub PAT expired or revoked")
            return CheckResult.fail(ErrorReason.NO_ACCESS, "Insufficient permissions")
        elif code == 404:
            return CheckResult.fail(ErrorReason.NO_ACCESS)
        elif code >= 500:
            return CheckResult.fail(ErrorReason.SERVER_ERROR, "GitHub API server error")
        return CheckResult.fail(ErrorReason.UNKNOWN, f"Unexpected response code: {code}")

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        headers = self._headers(token)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST, "Missing token")
        url = f"{self._base_url}/user"

        # Robust retry for transient TLS/EOF/network issues
        last_err: Optional[str] = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    code = resp.getcode()
                    content = resp.read().decode("utf-8", "ignore")
                    return self._judge(code, content)
            except HTTPError as e:
                code = getattr(e, "code", 0) or 0
                try:
                    content = e.read().decode("utf-8", "ignore")
                except Exception:
                    content = str(e)
                return self._judge(code, content)
            except URLError as e:
                last_err = str(e)
                # Treat as transient, backoff and retry
                time.sleep(0.8 * (attempt + 1))
            except Exception as e:
                last_err = str(e)
                time.sleep(0.8 * (attempt + 1))

        if last_err:
            logger.error(f"GitHub API request failed after retries: {last_err}")
        # Classify as rate-limited/transient to allow WAIT_CHECK retries by pipeline
        return CheckResult.fail(ErrorReason.RATE_LIMITED, f"Transient failure: {last_err or 'unknown'}")

    @handle_exceptions(default_result=[], log_level="warning")
    def _fetch_user_repos(self, url: str, headers: Dict) -> List[str]:
        if not url:
            return []
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.getcode() != 200:
                    return []
                content = resp.read().decode("utf-8", "ignore")
                data = json.loads(content)
                if isinstance(data, list):
                    return [trim(x.get("full_name", "")) for x in data if x.get("full_name")]
        except Exception as e:
            logger.warning(f"Failed to fetch GitHub repositories: {e}")
        return []

    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        headers = self._headers(token)
        if not headers:
            return []
        url = f"{self._base_url}/user/repos?per_page=30&sort=updated"
        return self._fetch_user_repos(url, headers)


register_provider("github", GitHubProvider)

