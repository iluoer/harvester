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

from constant.system import CTX, NO_RETRY_ERROR_CODES
from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.coordinator import get_user_agent
from tools.logger import get_logger
from tools.utils import trim

from ..client import urlopen
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
                    with urlopen(req, timeout=timeout, context=CTX) as response:
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
                    if not isinstance(e, urllib.error.URLError) or not isinstance(e.reason, socket.timeout):
                        logger.error(f"Check Claude session error, key: {token}, message: {traceback.format_exc()}")

                attempt += 1
                time.sleep(1)

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

        return super().check(token=token, address=address, endpoint=endpoint, model=model)

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge Anthropic API response."""
        message = trim(message)
        if re.findall(r"credit balance is too low|Billing|purchase", message, flags=re.I):
            return CheckResult.fail(ErrorReason.NO_QUOTA)
        elif code == 404 and re.findall(r"not_found_error", trim(message), flags=re.I):
            return CheckResult.fail(ErrorReason.NO_MODEL)

        return super()._judge(code, message)

    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available Anthropic models."""
        token = trim(token)
        if not token:
            return []

        # see: https://docs.anthropic.com/en/docs/about-claude/models
        return [
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


register_provider("anthropic", AnthropicProvider)
