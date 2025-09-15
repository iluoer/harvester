#!/usr/bin/env python3

"""
Doubao provider implementation.
"""

import urllib.parse
from typing import List

from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.utils import trim

from ..client import chat
from .openai_like import OpenAILikeProvider
from .registry import register_provider


class DoubaoProvider(OpenAILikeProvider):
    """Doubao provider implementation."""

    def __init__(self, conditions: List[Condition], **kwargs):
        # Set Doubao specific defaults
        self.defaults(
            kwargs,
            {
                "name": "doubao",
                "base_url": "https://ark.cn-beijing.volces.com",
                "completion_path": "/api/v3/chat/completions",
                "model_path": "/api/v3/models",
                "default_model": "doubao-pro-32k",
                "model_pattern": r"ep-[0-9]{14}-[a-z0-9]{5}",
            },
        )

        super().__init__(conditions=conditions, **kwargs)

    def _judge(self, code: int, message: str) -> CheckResult:
        """Judge Doubao API response."""
        if code == 404:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        return super()._judge(code, message)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check Doubao token validity.

        Doubao (Volcengine Ark) can use either the model name or the EndpointId (ep-...)
        as the "model" parameter. Prefer the provided EndpointId when available and
        also attach it as X-Endpoint-Id header to maximize compatibility across SDKs.
        """
        # Build headers with optional endpoint id
        additional = {}
        ep = trim(endpoint)
        if ep:
            additional["X-Endpoint-Id"] = ep
        headers = self._get_headers(token=token, additional=additional)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        # Resolve URL and model: use endpoint id if provided, else fallback to default
        chosen_model = trim(model) or (ep if ep else self._default_model)
        url = urllib.parse.urljoin(self._base_url, self.completion_path)

        code, message = chat(url=url, headers=headers, model=chosen_model)
        return self._judge(code=code, message=message)

    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available Doubao models for this token/endpoint scope.

        Attaches X-Endpoint-Id when provided to scope model listing. Best-effort.
        """
        additional = {}
        ep = trim(endpoint)
        if ep:
            additional["X-Endpoint-Id"] = ep
        headers = self._get_headers(token=token, additional=additional)
        if not headers:
            return []
        url = urllib.parse.urljoin(self._base_url, self.model_path)
        return self._fetch_models(url=url, headers=headers)


register_provider("doubao", DoubaoProvider)
