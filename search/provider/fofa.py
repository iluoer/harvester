#!/usr/bin/env python3

"""
FOFA provider implementation for harvesting and validating FOFA credentials.
Validation strategy:
- Uses the FOFA API to verify email+key pair by calling a lightweight endpoint.
- Endpoint: https://fofa.info/api/v1/info/my?email=<email>&key=<key>
  Returns JSON with user info when valid; returns error when invalid.

Search/collect still runs through the generic pipeline; this provider focuses on check().
"""

from __future__ import annotations

import json
import random
import time
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


class FofaProvider(AIBaseProvider):
    """FOFA provider.

    For FOFA, token is the API key (32 hex). Email can be paired from extraction and
    is passed via Service.endpoint or extras if not found.
    """

    def __init__(self, conditions: List[Condition], **kwargs):
        # FOFA does not use base_url/completion_path/model_path; provide placeholders
        self.defaults(
            kwargs,
            {
                "name": "fofa",
                "base_url": "https://fofa.info",
                "completion_path": "/api/v1/info/my",
                "model_path": "",
                "default_model": "default",
            },
        )

        # Allow passing a default email via extras
        self._default_email = trim(kwargs.get("email", ""))

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
                "email",
            ]),
        )

    def _get_headers(self, token: str, additional: dict | None = None) -> dict | None:
        # FOFA uses query params instead of Authorization header
        # Still return a minimal header for HTTP request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Connection": "keep-alive",
            "Referer": "https://fofa.info/",
            "Origin": "https://fofa.info",
        }
        if isinstance(additional, dict):
            headers.update(additional)
        return headers

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        key = trim(token)
        if not key:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        # FOFA 支持仅 key 校验；若能配对到 email 则一并传递
        email = trim(endpoint) or self._default_email

        try:
            # Build URL: https://fofa.info/api/v1/info/my?key=... [&email=...]
            base = trim(self._base_url) or "https://fofa.info"
            path = trim(self.completion_path) or "/api/v1/info/my"
            url = urllib.parse.urljoin(base + ("/" if not base.endswith("/") else ""), path.lstrip("/"))
            params = {"key": key}
            if email:
                params["email"] = email
            full_url = url + "?" + urllib.parse.urlencode(params)

            # Robust fetch with light retries when 200 but body is not JSON
            headers = self._get_headers(token)
            attempts = 0
            while True:
                attempts += 1
                req = urllib.request.Request(url=full_url, headers=headers)
                with urllib.request.urlopen(req, timeout=25) as resp:
                    code = resp.getcode()
                    body = resp.read().decode("utf-8", "ignore")
                    if code == 200:
                        try:
                            data = json.loads(body)
                        except Exception:
                            # 200 but not JSON: retry a few times with jitter to bypass anti-bot
                            if attempts <= 5:
                                time.sleep(0.4 + random.random() * 0.8)
                                continue
                            return CheckResult.fail(ErrorReason.NO_ACCESS, message="non-json response", status_code=200)

                        # Explicit failure case
                        if isinstance(data, dict) and data.get("error") is True:
                            msg = str(data.get("errmsg", ""))
                            lowered = msg.lower()
                            if "invalid" in lowered or "wrong" in lowered:
                                return CheckResult.fail(ErrorReason.INVALID_KEY, message=msg, status_code=200)
                            # If error message hints missing email pairing, treat as transient access issue to allow retries/pairing
                            if "email" in lowered or "mail" in lowered:
                                return CheckResult.fail(ErrorReason.NO_ACCESS, message=msg or "fofa email missing", status_code=200)
                            return CheckResult.fail(
                                ErrorReason.BAD_REQUEST,
                                message=msg or "fofa key requires pairing",
                                status_code=200,
                            )
                        # Success: 尝试提取会员/额度信息用于附加说明
                        msg_bits = []
                        if isinstance(data, dict):
                            for k in ("email", "mail", "user"):
                                if data.get(k):
                                    msg_bits.append(f"email={data.get(k)}")
                                    break
                            if data.get("isvip") is not None:
                                msg_bits.append(f"isvip={data.get('isvip')}")
                            for k in ("vip_level", "vip", "level"):
                                if data.get(k) is not None:
                                    msg_bits.append(f"vip_level={data.get(k)}")
                                    break
                            for k in ("fcoin", "foin", "coin"):
                                if data.get(k) is not None:
                                    msg_bits.append(f"coin={data.get(k)}")
                                    break
                        return CheckResult.success(message=("; ".join(msg_bits) or "FOFA key ok"))
                    # Non-200: attempt simple classification
                    if code == 401:
                        return CheckResult.fail(ErrorReason.INVALID_KEY, status_code=code)
                    if code == 403:
                        return CheckResult.fail(ErrorReason.NO_ACCESS, status_code=code)
                    if code == 429:
                        return CheckResult.fail(ErrorReason.RATE_LIMITED, status_code=code)
                    if code >= 500:
                        return CheckResult.fail(ErrorReason.SERVER_ERROR, status_code=code)
                    return CheckResult.fail(ErrorReason.UNKNOWN, status_code=code)
                # Non-200: attempt simple classification
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
            logger.error(f"FOFA check error: {e}")
            return CheckResult.fail(ErrorReason.UNKNOWN, message=str(e))


# Register provider type
register_provider("fofa", FofaProvider)
