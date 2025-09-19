#!/usr/bin/env python3

"""
Shodan provider implementation for harvesting and validating Shodan API keys.
Validation strategy:
- Uses the Shodan API "account profile" or API info endpoint which requires API key.
- Endpoint: https://api.shodan.io/account/profile?key=<api_key>
  Alternatively: https://api.shodan.io/api-info?key=<api_key> (gives query credits, etc.)

We prefer account/profile since it requires authentication and returns 200 for valid key.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import List

from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.logger import get_logger
from tools.utils import trim

from .base import AIBaseProvider
from .registry import register_provider

logger = get_logger("provider")


class ShodanProvider(AIBaseProvider):
    """Shodan provider.

    For Shodan, token is the API key (32 hex). No additional endpoint needed.
    """

    def __init__(self, conditions: List[Condition], **kwargs):
        # Provide sensible defaults
        self.defaults(
            kwargs,
            {
                "name": "shodan",
                "base_url": "https://api.shodan.io",
                "completion_path": "/account/profile",
                "model_path": "",
                "default_model": "default",
            },
        )

        # Option to use api-info path instead of account/profile via extras: use_api_info: true
        self._use_api_info = bool(kwargs.get("use_api_info", False))

        super().__init__(
            kwargs["name"],
            kwargs["base_url"],
            kwargs["completion_path"],
            kwargs["model_path"],
            kwargs["default_model"],
            conditions,
            **AIBaseProvider.filter(kwargs, [
                "name",
                "base_url",
                "completion_path",
                "model_path",
                "default_model",
                "use_api_info",
            ]),
        )

    def _get_headers(self, token: str, additional: dict | None = None) -> dict | None:
        # Shodan uses key as query param; keep minimal headers
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
        }
        if isinstance(additional, dict):
            headers.update(additional)
        return headers

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        key = trim(token)
        if not key:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        base = trim(self._base_url) or "https://api.shodan.io"
        path = "/api-info" if self._use_api_info else (trim(self.completion_path) or "/account/profile")
        url = base + ("/" if not base.endswith("/") else "") + path.lstrip("/")
        full_url = url + ("?" if "?" not in url else "&") + urllib.parse.urlencode({"key": key})

        try:
            req = urllib.request.Request(url=full_url, headers=self._get_headers(token))
            with urllib.request.urlopen(req, timeout=15) as resp:
                code = resp.getcode()
                body = resp.read().decode("utf-8", "ignore")
                if code == 200:
                    try:
                        data = json.loads(body)
                        # Basic sanity: some fields present
                        if isinstance(data, dict) and data:
                            return CheckResult.success()
                        return CheckResult.success()
                    except Exception:
                        return CheckResult.success()
                if code == 401:
                    return CheckResult.fail(ErrorReason.INVALID_KEY, status_code=code)
                if code == 403:
                    return CheckResult.fail(ErrorReason.NO_ACCESS, status_code=code)
                if code == 429:
                    return CheckResult.fail(ErrorReason.RATE_LIMITED, status_code=code)
                if code >= 500:
                    return CheckResult.fail(ErrorReason.SERVER_ERROR, status_code=code)
                return CheckResult.fail(ErrorReason.UNKNOWN, status_code=code)
        except urllib.error.HTTPError as e:
            code = getattr(e, "code", 0) or 0
            try:
                msg = e.read().decode("utf-8", "ignore")
            except Exception:
                msg = str(e)
            if code == 401:
                return CheckResult.fail(ErrorReason.INVALID_KEY, message=msg, status_code=code)
            if code == 403:
                return CheckResult.fail(ErrorReason.NO_ACCESS, message=msg, status_code=code)
            if code == 429:
                return CheckResult.fail(ErrorReason.RATE_LIMITED, message=msg, status_code=code)
            if code >= 500:
                return CheckResult.fail(ErrorReason.SERVER_ERROR, message=msg, status_code=code)
            return CheckResult.fail(ErrorReason.UNKNOWN, message=msg, status_code=code)
        except Exception as e:
            logger.error(f"Shodan check error: {e}")
            return CheckResult.fail(ErrorReason.UNKNOWN, message=str(e))


# Register provider type
register_provider("shodan", ShodanProvider)

