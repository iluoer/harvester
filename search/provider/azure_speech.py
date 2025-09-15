#!/usr/bin/env python3

"""
Azure Speech (Cognitive Services) provider implementation.
- Focus: Speech-to-Text/Speech token validation via STS issueToken endpoint
- Auth: Ocp-Apim-Subscription-Key header with region-specific base

This provider validates a candidate key by POSTing to:
  https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken
If a direct address is provided (api.cognitive.microsoft.com or stt.speech.microsoft.com),
it derives the correct STS endpoint automatically.
"""

from typing import Dict, List, Optional
import re
import urllib.request
import urllib.error
import urllib.parse

from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.coordinator import get_user_agent
from tools.logger import get_logger
from tools.utils import trim

from .base import AIBaseProvider
from .registry import register_provider

logger = get_logger("provider")


class AzureSpeechProvider(AIBaseProvider):
    """Azure Cognitive Services Speech provider."""

    def __init__(self, conditions: List[Condition], **kwargs):
        # Defaults (base_url used only as fallback)
        self.defaults(
            kwargs,
            {
                "name": "azure_speech",
                "base_url": "https://eastus.api.cognitive.microsoft.com/",
                "completion_path": "sts/v1.0/issueToken",
                "model_path": "",  # not used
                "default_model": "eastus",  # overload to carry default region
            },
        )

        # Avoid passing duplicate named args to base __init__
        passthrough = AIBaseProvider.filter(
            kwargs,
            [
                "name",
                "base_url",
                "completion_path",
                "model_path",
                "default_model",
            ],
        )

        super().__init__(
            kwargs["name"],
            kwargs["base_url"],
            kwargs["completion_path"],
            kwargs["model_path"],
            kwargs["default_model"],
            conditions,
            **passthrough,
        )

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        token = trim(token)
        if not token:
            return None
        headers = {
            "Ocp-Apim-Subscription-Key": token,
            # issueToken accepts empty body; use form content-type to be safe
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "User-Agent": get_user_agent(),
            "Accept": "application/json, text/plain, */*",
        }
        if isinstance(additional, dict):
            headers.update(additional)
        return headers

    def _judge(self, code: int, message: str) -> CheckResult:
        # Speech STS returns 200 with a JWT token string in body when key is valid
        if code == 200:
            return CheckResult.success()
        if code == 401:
            return CheckResult.fail(ErrorReason.INVALID_KEY)
        if code == 403:
            return CheckResult.fail(ErrorReason.NO_ACCESS)
        if code == 429:
            return CheckResult.fail(ErrorReason.RATE_LIMITED)
        if code >= 500:
            return CheckResult.fail(ErrorReason.SERVER_ERROR)
        return CheckResult.fail(ErrorReason.UNKNOWN)

    def __candidate_sts_urls(self, address: str = "", region: str = "") -> List[str]:
        """Return candidate STS token endpoints for Speech/Cognitive Services.
        Accepts either:
          - address like https://eastus.api.cognitive.microsoft.com or https://eastus.stt.speech.microsoft.com
          - region string like eastus/eastus2/westeurope
        We try multiple well-known endpoints to maximize validation coverage.
        """
        addr = trim(address).removesuffix("/")
        reg = trim(region)
        candidates: List[str] = []

        # If address provided, try to parse host and derive best candidates
        host = ""
        if addr:
            try:
                parsed = urllib.parse.urlparse(addr)
                host = parsed.hostname or ""
            except Exception:
                host = ""

            # Resource-based endpoint (resource.cognitiveservices.azure.com)
            m_host = re.match(r"^([a-z0-9-]+)\.cognitiveservices\.azure\.com$", host or "", flags=re.I)
            if m_host:
                resource = m_host.group(1)
                # Token endpoints for resource-based hosts
                candidates.append(f"https://{resource}.cognitiveservices.azure.com/sts/v1.0/issueToken")
                candidates.append(f"https://{resource}.cognitiveservices.azure.com/cognitiveservices/token")
            else:
                # Region-based hosts
                m = re.search(r"https?://([a-z0-9-]+)\.(?:api\.cognitive\.microsoft\.com|stt\.speech\.microsoft\.com)", addr, flags=re.I)
                if m:
                    reg = reg or m.group(1)

        # Fallback to configured default if region still unknown
        if not reg:
            reg = trim(self._default_model) if self._default_model else "eastus"

        # Region-based STS candidates
        if re.match(r"^[a-z0-9-]{2,}$", reg, flags=re.I):
            candidates.append(f"https://{reg}.api.cognitive.microsoft.com/sts/v1.0/issueToken")
            candidates.append(f"https://{reg}.sts.speech.microsoft.com/cognitiveservices/token")

        # De-duplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for u in candidates:
            if u not in seen:
                uniq.append(u)
                seen.add(u)
        return uniq

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        token = trim(token)
        if not token:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        # Region can come from endpoint/model (we overload to carry SPEECH_REGION)
        region = trim(endpoint) or trim(model)
        urls = self.__candidate_sts_urls(address=address, region=region)
        if not urls:
            logger.error(f"Invalid STS URLs derived from address='{address}', region='{region}'")
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        headers = self._get_headers(token=token)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        # Minimal POST with empty body
        data = b""  # issueToken expects a POST; empty body is acceptable

        # Try candidates until one succeeds; aggregate most meaningful failure
        last_result: Optional[CheckResult] = None
        for url in urls:
            req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    code = resp.getcode()
                    result = self._judge(code=code, message="")
                    if result.ok:
                        return result
                    last_result = result
            except urllib.error.HTTPError as e:
                code = e.code
                try:
                    msg = e.read().decode("utf-8", errors="ignore")
                except Exception:
                    msg = e.reason
                result = self._judge(code=code, message=trim(msg))
                # If any 200 happens we would have returned; otherwise keep the most definitive error
                # Prefer INVALID over others, then NO_ACCESS, then RATE_LIMITED, then SERVER_ERROR, then UNKNOWN
                if not last_result or result.reason in (ErrorReason.INVALID_KEY, ErrorReason.NO_ACCESS, ErrorReason.RATE_LIMITED, ErrorReason.SERVER_ERROR):
                    last_result = result
            except Exception as e:
                logger.warning(f"Speech check error for {url}: {e}")
                # Keep trying other candidates; remember at least UNKNOWN
                if not last_result:
                    last_result = CheckResult.fail(ErrorReason.UNKNOWN)

        return last_result or CheckResult.fail(ErrorReason.UNKNOWN)

    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        # For Speech, we don't enumerate models; optionally, we could return regions
        return []


# Register provider type
register_provider("azure_speech", AzureSpeechProvider)

