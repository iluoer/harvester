#!/usr/bin/env python3

"""
Anthropic provider implementation.
"""

import json
import re
import socket
import time
import traceback
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from constant.system import CTX, NO_RETRY_ERROR_CODES, DEFAULT_QUESTION
from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.coordinator import get_user_agent
from tools.logger import get_logger
from tools.utils import trim

from .base import AIBaseProvider
from .registry import register_provider

logger = get_logger("provider")


class AnthropicProvider(AIBaseProvider):
    """Anthropic provider implementation."""

    def __init__(self, conditions: List[Condition], **kwargs):
        # Extract parameters with defaults
        config = self.extract(
            kwargs,
            {
                "name": "anthropic",
                "base_url": "https://api.anthropic.com",
                "completion_path": "/v1/messages",
                "model_path": "",
                "default_model": "claude-sonnet-4-20250514",
            },
        )

        super().__init__(
            config["name"],
            config["base_url"],
            config["completion_path"],
            config["model_path"],
            config["default_model"],
            conditions,
            **kwargs,
        )

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        """Get headers for Anthropic API requests."""
        token = trim(token)
        if not token:
            return None

        return {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        }

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check Anthropic token validity."""
        token = trim(token)
        if token.startswith("sk-ant-sid01-"):
            # Handle Claude session tokens
            url = "https://api.claude.ai/api/organizations"
            headers = {
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "max-age=0",
                "cookie": f"sessionKey={token}",
                "user-agent": get_user_agent(),
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
            }

            content, success = "", False
            attempt, retries, timeout = 0, 3, 10

            req = urllib.request.Request(url, headers=headers, method="GET")
            while attempt < retries:
                try:
                    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as response:
                        content = response.read().decode("utf8")
                        success = True
                        break
                except urllib.error.HTTPError as e:
                    if e.code == 401:
                        return CheckResult.fail(ErrorReason.INVALID_KEY)
                    else:
                        try:
                            content = e.read().decode("utf8")
                            if not content.startswith("{") or not content.endswith("}"):
                                content = e.reason
                        except:
                            content = e.reason

                        if e.code == 403:
                            message = ""
                            try:
                                data = json.loads(content)
                                message = data.get("error", {}).get("message", "")
                            except:
                                message = content

                            if re.findall(r"Invalid authorization", message, flags=re.I):
                                return CheckResult.fail(ErrorReason.INVALID_KEY)

                        if e.code in NO_RETRY_ERROR_CODES:
                            break
                except Exception as e:
                    # Retry on transient network issues including SSL EOF
                    transient = isinstance(e, urllib.error.URLError) or isinstance(getattr(e, 'reason', None), socket.timeout)
                    if not transient:
                        logger.error(f"Check Claude session error, key: {token}, message: {traceback.format_exc()}")

                attempt += 1
                # Exponential backoff with jitter
                time.sleep(min(3, 0.5 * (2 ** attempt)) + (0.2 * (attempt % 2)))

            if not content or re.findall(r"Invalid authorization", content, flags=re.I):
                return CheckResult.fail(ErrorReason.INVALID_KEY)
            elif not success:
                logger.error(f"Check Claude session error, key: {token}, message: {content}")
                return CheckResult.fail(ErrorReason.UNKNOWN)

            try:
                data = json.loads(content)
                valid = False
                if data and isinstance(data, list):
                    valid = trim(data[0].get("name", None)) != ""

                    capabilities = data[0].get("capabilities", [])
                    if capabilities and isinstance(capabilities, list) and "claude_pro" in capabilities:
                        logger.info(f"Found Claude Pro key: {token}")

                if not valid:
                    logger.warning(f"Check error, Anthropic session key: {token}, message: {content}")

                return CheckResult.success() if valid else CheckResult.fail(ErrorReason.INVALID_KEY)
            except:
                return CheckResult.fail(ErrorReason.INVALID_KEY)

        # For API keys (not sid01), call /v1/messages with Anthropic payload shape
        token = trim(token)
        headers = self._get_headers(token=token)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        primary = trim(model) or self._default_model
        url = urllib.parse.urljoin(self._base_url, self.completion_path)

        candidates = [m for m in [primary, "claude-3-5-haiku-latest", "claude-3-haiku-20240307"] if m]
        last_result = None
        for m in candidates:
            params = {
                "model": m,
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": DEFAULT_QUESTION}]}
                ],
            }

            try:
                payload = json.dumps(params).encode("utf-8")
                req = urllib.request.Request(url=url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10, context=CTX) as resp:
                    _ = resp.read().decode("utf-8", "ignore")
                    return CheckResult.success()
            except urllib.error.HTTPError as e:
                code = e.code
                try:
                    body = e.read().decode("utf-8", "ignore")
                except Exception:
                    body = e.reason
                msg = trim(body)
                if code == 401 or re.findall(r"invalid.*api[_-]?key", msg, flags=re.I):
                    return CheckResult.fail(ErrorReason.INVALID_KEY, message=msg, status_code=code)
                # Treat permission denied (403) as valid credentials but no access to this action
                if code == 403 and re.findall(r"permission[_-]?error|not authorized|forbidden", msg, flags=re.I):
                    return CheckResult.success()
                # Treat low credit/billing issues (402) as valid credentials but no quota
                if code == 402 or re.findall(r"credit balance is too low|billing|purchase", msg, flags=re.I):
                    return CheckResult.success()
                if code == 429:
                    last_result = CheckResult.fail(ErrorReason.RATE_LIMITED, message=msg, status_code=code)
                    break
                if code >= 500:
                    last_result = CheckResult.fail(ErrorReason.SERVER_ERROR, message=msg, status_code=code)
                    continue
                # If model not found, try next candidate
                if code == 404 and re.findall(r"not_found_error", msg, flags=re.I):
                    last_result = CheckResult.fail(ErrorReason.NO_MODEL, message=msg, status_code=code)
                    continue
                last_result = CheckResult.fail(ErrorReason.UNKNOWN, message=msg, status_code=code)
            except Exception as e:
                logger.debug(f"Anthropic check transient error for model {m}: {e}")
                last_result = CheckResult.fail(ErrorReason.UNKNOWN, message=str(e))
        return last_result or CheckResult.fail(ErrorReason.UNKNOWN)

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge Anthropic API response."""
        message = trim(message)
        if re.findall(r"credit balance is too low|Billing|purchase", message, flags=re.I):
            return CheckResult.fail(ErrorReason.NO_QUOTA)
        elif code == 404 and re.findall(r"not_found_error", trim(message), flags=re.I):
            return CheckResult.fail(ErrorReason.NO_MODEL)

        return super()._judge(code, message)

    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """Return model markers and, for session tokens, paid capability if detectable.

        - Always return a models=... marker with a conservative model list (static reference)
        - If token is a session (sk-ant-sid01-), query organizations to detect 'claude_pro'
        """
        token = trim(token)
        if not token:
            return []

        markers: List[str] = []
        # Conservative static list (docs reference)
        models = [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]
        try:
            markers.append("models=" + ",".join(models))
        except Exception:
            pass

        # If it's a Claude session token, try to infer paid capability
        if token.startswith("sk-ant-sid01-"):
            url = "https://api.claude.ai/api/organizations"
            headers = {
                "accept": "application/json, text/plain, */*",
                "cookie": f"sessionKey={token}",
                "user-agent": get_user_agent(),
            }
            req = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=10, context=CTX) as resp:
                    content = resp.read().decode("utf-8", "ignore")
                try:
                    data = json.loads(content)
                    if isinstance(data, list) and data:
                        caps = data[0].get("capabilities", [])
                        if isinstance(caps, list):
                            paid = "claude_pro" in caps
                            markers.append(f"paid={'true' if paid else 'false'}")
                except Exception:
                    pass
            except Exception:
                # Silent best-effort; do not break inspect
                pass

        return markers


register_provider("anthropic", AnthropicProvider)
