#!/usr/bin/env python3

"""
OpenAI-like provider base class.
"""

import json
import re
import urllib.parse

from tools.logger import get_logger

logger = get_logger("provider")
from typing import Dict, List, Optional

from constant.system import DEFAULT_COMPLETION_PATH, DEFAULT_MODEL_PATH
from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.coordinator import get_user_agent
from tools.utils import handle_exceptions, trim

from ..client import http_get, chat
from .base import AIBaseProvider
from .registry import register_provider


class OpenAILikeProvider(AIBaseProvider):
    """Base class for OpenAI-compatible providers."""

    def __init__(self, conditions: List[Condition], **kwargs):
        # Extract required parameters without defaults
        name = trim(kwargs.pop("name", ""))
        base_url = trim(kwargs.pop("base_url", ""))
        default_model = trim(kwargs.pop("default_model", ""))

        # Validate required parameters
        if not name:
            raise ValueError("OpenAILike provider requires 'name' parameter to be specified")
        if not base_url:
            raise ValueError(f"OpenAILike provider {name} requires 'base_url' parameter to be specified")
        if not default_model:
            raise ValueError(f"OpenAILike provider {name} requires 'default_model' parameter to be specified")

        # Extract optional parameters with defaults
        config = self.extract(
            kwargs,
            {
                "completion_path": DEFAULT_COMPLETION_PATH,
                "model_path": DEFAULT_MODEL_PATH,
            },
        )

        # Add the validated required parameters back to config
        config.update(
            {
                "name": name,
                "base_url": base_url,
                "default_model": default_model,
            }
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
        """Get headers for OpenAI-like API requests."""
        token = trim(token)
        if not token:
            return None

        if not isinstance(additional, dict):
            additional = {}

        auth_key = (trim(self.extras.get("auth_key", None)) if isinstance(self.extras, dict) else "") or "authorization"
        auth_value = f"Bearer {token}" if auth_key.lower() == "authorization" else token

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            auth_key: auth_value,
            "user-agent": get_user_agent(),
        }
        headers.update(additional)

        return headers

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge OpenAI-like API response."""
        if code == 200:
            try:
                data = json.loads(trim(message))
                if data and isinstance(data, dict):
                    error = data.get("error", None)
                    if error and isinstance(error, dict):
                        error_type = trim(error.get("type", ""))
                        error_reason = trim(error.get("message", "")).lower()

                        if error_type or "authorization" in error_reason:
                            return CheckResult.fail(ErrorReason.INVALID_KEY)
            except:
                logger.error(f"Failed to parse response, domain: {self._base_url}, message: {message}")
                return CheckResult.fail(ErrorReason.UNKNOWN)

            return CheckResult.success()

        message = trim(message)
        if message:
            if code == 403:
                # 余额/欠费：标记 NO_QUOTA（覆盖 Doubao 的 AccountOverdueError 等）
                if re.findall(r"AccountOverdueError|overdue\s+balance|欠费|余额不足|need\s*recharge", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_QUOTA)
                # 权限不足/被禁止：视为存在键但不可用，标记 NO_ACCESS
                if re.findall(r"unauthorized|已被封禁|forbidden|permission", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_ACCESS)
                # 模型不存在：标记 NO_MODEL
                if re.findall(r"model_not_found", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_MODEL)
                # 区域/权限不匹配：标记 NO_ACCESS
                if re.findall(r"unsupported_country_region_territory|该令牌无权访问模型", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_ACCESS)
                # 配额/额度限制：标记 NO_QUOTA
                if re.findall(r"exceeded_current_quota_error|insufficient_user_quota|insufficient_quota", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_QUOTA)
            elif code == 429:
                if re.findall(r"insufficient_quota|billing_not_active|欠费|请充值|recharge", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.NO_QUOTA)
                if re.findall(r"rate_limit_exceeded", message, flags=re.I):
                    return CheckResult.fail(ErrorReason.RATE_LIMITED)
            elif code == 503 and re.findall(r"无可用渠道", message, flags=re.I):
                return CheckResult.fail(ErrorReason.NO_MODEL)

        return super()._judge(code, message)

    @handle_exceptions(default_result=[], log_level="warning")
    def _fetch_models(self, url: str, headers: Dict) -> List[str]:
        """Fetch models from API endpoint."""
        url = trim(url)
        if not url:
            return []

        content = http_get(url=url, headers=headers, interval=1)
        if not content:
            return []

        result = json.loads(content)
        return [trim(x.get("id", "")) for x in result.get("data", [])]

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Override to choose a sane URL.

        - If an address was extracted but its host does not match our base_url host, ignore it
          and use base_url + completion_path to avoid malformed/foreign URLs causing 4xx.
        - Otherwise, use the provided address.
        """
        # Determine candidate URL
        base = trim(self._base_url)
        path = trim(self.completion_path)
        addr = trim(address)
        url = ""
        try:
            if addr and addr.lower().startswith(("http://", "https://")):
                base_host = urllib.parse.urlparse(base).netloc if base else ""
                addr_host = urllib.parse.urlparse(addr).netloc
                if base_host and addr_host and base_host not in addr_host:
                    # Host mismatch -> fallback to base
                    url = urllib.parse.urljoin(base, path)
                else:
                    url = addr
            else:
                url = urllib.parse.urljoin(base, path)
        except Exception:
            url = urllib.parse.urljoin(base, path)

        headers = self._get_headers(token=token)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)
        m = trim(model) or self._default_model
        code, message = chat(url=url, headers=headers, model=m)
        return self._judge(code=code, message=message)

    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available models from OpenAI-like API."""
        headers = self._get_headers(token=token)
        if not headers or not self._base_url or not self.model_path:
            return []

        url = urllib.parse.urljoin(self._base_url, self.model_path)
        return self._fetch_models(url=url, headers=headers)


register_provider("openai_like", OpenAILikeProvider)
