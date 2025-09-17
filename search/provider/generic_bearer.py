#!/usr/bin/env python3

"""
Generic Bearer/Token provider for simple key validation.
Used for speech providers exposing an account/me endpoint that returns 200 when
provided a valid API key.

Configurable via kwargs:
- name: provider name
- base_url: optional base, not strictly used (we take address or validation_urls)
- validation_urls: list[str] of absolute URLs to try in order
- auth_scheme: header auth scheme, e.g., "Bearer" (default) or "Token"
- header_name: header key for token, default: "Authorization"
- timeout: request timeout seconds (default 15)
- retries: not used here (handled by outer framework if any)

We only issue a simple GET request with proper header and judge on HTTP code.
"""

from typing import Dict, List, Optional
import urllib.request
import urllib.error
import urllib.parse
import json

from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.coordinator import get_user_agent
from tools.logger import get_logger
from tools.utils import trim

from .base import AIBaseProvider
from .registry import register_provider

logger = get_logger("provider")


class GenericBearerTokenProvider(AIBaseProvider):
    """Provider that validates API keys using simple GET and Authorization header."""

    def __init__(self, conditions: List[Condition], **kwargs):
        # Defaults
        self.defaults(
            kwargs,
            {
                "name": "generic_bearer",
                "base_url": "",
                "completion_path": "",
                "model_path": "",
                "default_model": "default",
                "validation_urls": [],
                "auth_scheme": "Bearer",
                "header_name": "Authorization",
                "timeout": 15,
            },
        )

        passthrough = AIBaseProvider.filter(
            kwargs,
            [
                "name",
                "base_url",
                "completion_path",
                "model_path",
                "default_model",
                "validation_urls",
                "auth_scheme",
                "header_name",
                "timeout",
            ],
        )

        self._validation_urls: List[str] = kwargs.get("validation_urls", []) or []
        self._auth_scheme: str = trim(kwargs.get("auth_scheme", "Bearer")) or "Bearer"
        self._header_name: str = trim(kwargs.get("header_name", "Authorization")) or "Authorization"
        try:
            self._timeout: int = int(kwargs.get("timeout", 15))
        except Exception:
            self._timeout = 15

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
        value = f"{self._auth_scheme} {token}" if self._auth_scheme else token
        headers = {
            self._header_name: value,
            "User-Agent": get_user_agent(),
            "Accept": "application/json, text/plain, */*",
        }
        if isinstance(additional, dict):
            headers.update(additional)
        return headers

    def _judge(self, code: int, message: str) -> CheckResult:
        # For simple account/model endpoints, HTTP 200 implies valid token
        if code == 200:
            return CheckResult.success()
        # Treat clear auth failures as invalid
        if code == 401:
            return CheckResult.fail(ErrorReason.INVALID_KEY)
        # Distinguish 403 cases using message when possible
        if code == 403:
            m = (message or "").lower()
            if any(x in m for x in ["insufficient quota", "insufficient_quota", "billing", "欠费", "余额不足", "overdue", "quota"]):
                return CheckResult.fail(ErrorReason.NO_QUOTA)
            if any(x in m for x in ["forbidden", "not authorized", "permission", "no access"]):
                return CheckResult.fail(ErrorReason.NO_ACCESS)
            return CheckResult.fail(ErrorReason.INVALID_KEY)
        # Explicit no-quota semantics
        if code == 402:
            return CheckResult.fail(ErrorReason.NO_QUOTA)
        # 404 on account/me usually means endpoint not found, but if auth passed it wouldn't be 404. Mark as no access.
        if code == 404:
            return CheckResult.fail(ErrorReason.NO_ACCESS)
        if code == 429:
            return CheckResult.fail(ErrorReason.RATE_LIMITED)
        if code >= 500:
            return CheckResult.fail(ErrorReason.SERVER_ERROR)
        # Be permissive: consider other 4xx (e.g., 4xx with non-auth business errors) as unknown, to allow retries or alternate URLs
        return CheckResult.fail(ErrorReason.UNKNOWN)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        token = trim(token)
        if not token:
            return CheckResult.fail(ErrorReason.INVALID_KEY)

        headers = self._get_headers(token=token)
        if not headers:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        urls = list(self._validation_urls)
        # Allow address override if provided and looks like a URL
        a = trim(address)
        if a.startswith("http://") or a.startswith("https://"):
            urls = [a] + urls

        if not urls:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        last: Optional[CheckResult] = None
        for u in urls:
            req = urllib.request.Request(url=u, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    code = resp.getcode()
                    result = self._judge(code=code, message="")
                    if result.ok:
                        return result
                    last = result
            except urllib.error.HTTPError as e:
                code = e.code
                try:
                    body = e.read().decode("utf-8", "ignore")
                except Exception:
                    body = str(e.reason)
                result = self._judge(code=code, message=body)
                last = result
            except Exception as e:
                logger.warning(f"Generic bearer check error for {u}: {e}")
                if not last:
                    last = CheckResult.fail(ErrorReason.UNKNOWN)
        return last or CheckResult.fail(ErrorReason.UNKNOWN)

    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """List available models and attach optional account info (e.g., balance).

        Behavior:
        - If model_path is configured, GET base_url + model_path to list models.
        - If extras include `info_path` (or `info_url`), GET it and append balance-related
          markers like "balance=...", "chargeBalance=...", "totalBalance=...", "status=...".
        - Always prefer simple GET; never call completion endpoints here.
        """
        token = trim(token)
        headers = self._get_headers(token=token)
        if not headers:
            return []

        out: List[str] = []

        models: List[str] = []

        # 1) Models list (if configured)
        try:
            base = trim(self._base_url)
            path = trim(self.model_path)
            if base and path:
                url = urllib.parse.urljoin(base, path)
                req = urllib.request.Request(url=url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    body = resp.read().decode("utf-8", "ignore")
                try:
                    data = json.loads(body)
                    # SiliconFlow/OpenAI-like responses
                    if isinstance(data, dict) and isinstance(data.get("data"), list):
                        for x in data.get("data", []):
                            mid = x.get("id") or x.get("name")
                            if isinstance(mid, str) and mid:
                                models.append(mid)
                    if not models and isinstance(data, dict) and isinstance(data.get("models"), list):
                        for x in data.get("models", []):
                            mid = x.get("id") or x.get("name")
                            if isinstance(mid, str) and mid:
                                models.append(mid)
                except Exception:
                    # Non-JSON or unexpected format; ignore models
                    pass
        except Exception as e:
            logger.debug(f"inspect models fetch failed: {e}")

        # SiliconFlow: do not output models to avoid overly long valid entries
        allow_models_marker = self._name.lower() != "siliconflow"
        if models and allow_models_marker:
            out.append("models=" + ",".join(models))

        # 1.1) Heuristic paid flag based on models and configured markers
        # Skip this heuristic for SiliconFlow; we'll rely on balance info instead
        try:
            if allow_models_marker:
                paid_markers = []
                if isinstance(self.extras, dict):
                    pm = self.extras.get("paid_model_markers")
                    if isinstance(pm, list):
                        paid_markers = [trim(str(x)) for x in pm if trim(str(x))]
                if models and paid_markers:
                    paid_guess = any(m in paid_markers for m in models)
                    out.append(f"paid={'true' if paid_guess else 'false'}")
        except Exception:
            pass

        # 2) Optional account info (balance) — support multiple URLs
        try:
            base = trim(self._base_url)
            info_path = trim(self.extras.get("info_path", "")) if isinstance(self.extras, dict) else ""
            info_url_cfg = trim(self.extras.get("info_url", "")) if isinstance(self.extras, dict) else ""
            info_urls = []
            if isinstance(self.extras, dict):
                cfg_urls = self.extras.get("info_urls")
                if isinstance(cfg_urls, list):
                    for u in cfg_urls:
                        u = trim(str(u))
                        if u:
                            info_urls.append(u)
            info_url = ""
            if info_url_cfg:
                info_url = info_url_cfg
            elif base and info_path:
                info_url = urllib.parse.urljoin(base, info_path)
            if info_url:
                info_urls.insert(0, info_url)

            for u in info_urls:
                try:
                    req = urllib.request.Request(url=u, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                        body = resp.read().decode("utf-8", "ignore")
                    try:
                        data = json.loads(body)
                        payload = data.get("data", data)
                        # Try multiple typical keys
                        bal = str(payload.get("balance", payload.get("points", payload.get("quota", ""))))
                        cbal = str(payload.get("chargeBalance", payload.get("charged", "")))
                        tbal = str(payload.get("totalBalance", payload.get("total", "")))
                        status = str(payload.get("status", payload.get("plan", "")))
                        paid_val = None
                        try:
                            paid_val = float(payload.get("chargeBalance", payload.get("charged", 0))) > 0
                        except Exception:
                            paid_val = None
                        markers = []
                        if bal:
                            markers.append(f"balance={bal}")
                        if cbal:
                            markers.append(f"chargeBalance={cbal}")
                        if tbal:
                            markers.append(f"totalBalance={tbal}")
                        if status:
                            markers.append(f"status={status}")
                        if paid_val is not None:
                            markers.append(f"paid={'true' if paid_val else 'false'}")
                        if markers:
                            out.extend(markers)
                            break  # stop after first successful info URL
                    except Exception:
                        continue
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"inspect info fetch failed: {e}")

        return out


# Register provider type
register_provider("generic_bearer", GenericBearerTokenProvider)

