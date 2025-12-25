#!/usr/bin/env python3

"""
Google Gemini provider implementation.
"""

import json
import re
import urllib.parse

from tools.logger import get_logger

logger = get_logger("provider")
from typing import Dict, List, Optional

from constant.system import DEFAULT_QUESTION
from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.utils import trim

from ..client import chat, http_get
from .base import AIBaseProvider
from .registry import register_provider


class GeminiProvider(AIBaseProvider):
    """Google Gemini provider implementation."""

    def __init__(self, conditions: List[Condition], **kwargs):
        # Extract parameters with defaults
        config = self.extract(
            kwargs,
            {
                "name": "gemini",
                "base_url": "https://generativelanguage.googleapis.com",
                "completion_path": "/v1beta/models",
                "model_path": "/v1beta/models",
                "default_model": "gemini-2.5-pro",
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
        """Get headers for Gemini API requests."""
        return {"accept": "application/json", "content-type": "application/json"}

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge Gemini API response."""
        if code == 200:
            return CheckResult.success()

        message = trim(message)
        if code == 400:
            if re.findall(r"API_KEY_INVALID|API key expired", message, flags=re.I):
                return CheckResult.fail(ErrorReason.INVALID_KEY)
            elif re.findall(r"FAILED_PRECONDITION", message, flags=re.I):
                return CheckResult.fail(ErrorReason.NO_ACCESS)
        elif code == 403:
            if re.findall(r"PERMISSION_DENIED|Your API key was reported as leaked", message, flags=re.I):
                return CheckResult.fail(ErrorReason.INVALID_KEY)
        elif code == 429:
            if re.findall(r"RESOURCE_EXHAUSTED|Quota exceeded for quota metric", message, flags=re.I):
                return CheckResult.fail(ErrorReason.NO_QUOTA)

        return super()._judge(code, message)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check Gemini token validity."""
        token = trim(token)
        if not token:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        model = trim(model) or self._default_model
        url = f"{urllib.parse.urljoin(self._base_url, self.completion_path)}/{model}:generateContent?key={token}"

        params = {"contents": [{"role": "user", "parts": [{"text": DEFAULT_QUESTION}]}]}
        code, message = chat(url=url, headers=self._get_headers(token=token), params=params)
        return self._judge(code=code, message=message)

    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available Gemini models."""
        token = trim(token)
        if not token:
            return []

        url = urllib.parse.urljoin(self._base_url, self.model_path) + f"?key={token}"
        content = http_get(url=url, headers=self._get_headers(token=token), interval=1)
        if not content:
            return []

        try:
            data = json.loads(content)
            models = data.get("models", [])
            return [x.get("name", "").removeprefix("models/") for x in models]
        except:
            logger.error(f"Failed to parse models from response: {content}")
            return []


register_provider("gemini", GeminiProvider)
