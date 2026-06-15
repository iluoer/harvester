#!/usr/bin/env python3

"""
StabilityAI provider implementation.
"""

import codecs
import urllib.parse
import urllib.request

from tools.logger import get_logger

from .registry import register_provider

logger = get_logger("provider")
import time
import uuid
from typing import Dict, List, Optional, Tuple

from constant.system import CTX, NO_RETRY_ERROR_CODES
from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.coordinator import get_user_agent
from tools.utils import trim

from ..client import urlopen
from .base import AIBaseProvider


class StabilityAIProvider(AIBaseProvider):
    """StabilityAI provider implementation."""

    def __init__(self, conditions: List[Condition], **kwargs):
        # Extract parameters with defaults
        config = self.extract(
            kwargs,
            {
                "name": "stabilityai",
                "base_url": "https://api.stability.ai",
                "completion_path": "/v2beta/stable-image/generate",
                "model_path": "",
                "default_model": "core",
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
        """Get headers for StabilityAI API requests."""
        key = trim(token)
        if not key:
            return None

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "multipart/form-data",
            "Accept": "application/json",
        }
        if additional and isinstance(additional, dict):
            headers.update(additional)

        return headers

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        """Check StabilityAI token validity."""

        def post_multipart(
            url: str, token: str, fields: Optional[Dict] = None, files: Optional[Dict] = None, retries: int = 3
        ) -> Tuple[int, str]:
            url, token = trim(url), trim(token)
            if not url or not token:
                return 401, ""

            boundary, contents = str(uuid.uuid4()), []
            if not isinstance(fields, dict):
                fields = dict()
            if not isinstance(files, dict):
                files = dict()

            # add common form fields
            for k, v in fields.items():
                contents.append(f"--{boundary}")
                contents.append(f'Content-Disposition: form-data; name="{k}"')
                contents.append("Content-Type: text/plain")
                contents.append("")
                contents.append(v)
                contents.append("")

            # add files
            for k, v in files.items():
                filename, data = v
                contents.append(f"--{boundary}")
                contents.append(f'Content-Disposition: form-data; name="{k}"; filename="{filename}"')
                contents.append("Content-Type: application/octet-stream")
                contents.append("")
                contents.append(data)
                contents.append("")

            # add end flag
            contents.append(f"--{boundary}--")
            contents.append("")

            # encode content
            payload = b"\r\n".join(codecs.encode(x, encoding="utf8") for x in contents)

            req = urllib.request.Request(url, data=payload, method="POST")

            # set request headers
            req.add_header("Accept", "application/json")
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            req.add_header("User-Agent", get_user_agent())

            # send request with retry
            code, message, attempt, retries = 401, "", 0, max(1, retries)
            while attempt < retries:
                try:
                    with urlopen(req, timeout=15, context=CTX) as response:
                        code = 200
                        message = response.read().decode("utf8")
                        break
                except urllib.error.HTTPError as e:
                    code = e.code
                    if code != 401:
                        try:
                            message = e.read().decode("utf8")
                            if not message.startswith("{") or not message.endswith("}"):
                                message = e.reason
                        except:
                            message = e.reason

                        logger.error(
                            f"[chat] failed to request URL: {url}, token: {token}, status code: {code}, message: {message}"
                        )

                    if code in NO_RETRY_ERROR_CODES:
                        break
                except Exception:
                    pass

                attempt += 1
                time.sleep(1)

            return code, message

        token = trim(token)
        if not token:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        model = trim(model) or self._default_model
        url = f"{urllib.parse.urljoin(self._base_url, self.completion_path)}/{model}"
        fields = {"prompt": "Lighthouse on a cliff overlooking the ocean", "aspect_ratio": "3:2"}

        code, message = post_multipart(url=url, token=token, fields=fields)
        return self._judge(code=code, message=message)

    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available StabilityAI models."""
        return []


register_provider("stabilityai", StabilityAIProvider)
