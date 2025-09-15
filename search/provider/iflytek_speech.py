#!/usr/bin/env python3

"""
Iflytek (iFLYTEK/XFYun) Speech provider validation using WebAPI IAT REST endpoint.

We validate AppID + APIKey by calling the REST endpoint:
  POST https://iat-api.xfyun.cn/v2/iat
with headers:
  X-Appid: <appid>
  X-CurTime: <epoch seconds>
  X-Param: base64(json{"engine_type":"sms16k","aue":"raw"})
  X-CheckSum: md5(api_key + curTime + X-Param)

We send an empty audio field; if auth is correct, the API returns JSON with
non-auth error (e.g., missing audio) which we treat as success. If auth fails,
JSON contains error codes indicating illegal appid/apikey/sign.

Field mapping from Service -> check():
- token   => api_key
- model   => appid
- address => ignored
- endpoint=> ignored
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from typing import Dict, List, Optional
import urllib.parse
import urllib.request
import urllib.error

from core.enums import ErrorReason
from core.models import CheckResult, Condition
from tools.coordinator import get_user_agent
from tools.logger import get_logger
from tools.utils import trim

from .base import AIBaseProvider
from .registry import register_provider

logger = get_logger("provider")


class IflytekSpeechProvider(AIBaseProvider):
    def __init__(self, conditions: List[Condition], **kwargs):
        self.defaults(
            kwargs,
            {
                "name": "iflytek_speech",
                "base_url": "https://iat-api.xfyun.cn",
                "completion_path": "/v2/iat",
                "model_path": "",
                "default_model": "",
                "timeout": 20,
            },
        )

        # Extract required parameters for super().__init__
        name = kwargs["name"]
        base_url = kwargs["base_url"]
        completion_path = kwargs["completion_path"]
        model_path = kwargs["model_path"]
        default_model = kwargs["default_model"]

        # Filter out parameters that super().__init__ doesn't need
        filtered_kwargs = AIBaseProvider.filter(kwargs, [
            "name","base_url","completion_path","model_path","default_model","timeout"
        ])

        super().__init__(
            name,
            base_url,
            completion_path,
            model_path,
            default_model,
            conditions,
            **filtered_kwargs
        )

    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        # Not used; we build headers in check()
        return None

    def _build_headers(self, appid: str, api_key: str) -> Dict[str, str]:
        cur_time = str(int(time.time()))
        # Use standard params recommended by XFYun
        param_obj = {"engine_type": "sms16k", "aue": "raw"}
        param_b64 = base64.b64encode(json.dumps(param_obj, separators=(",", ":")).encode("utf-8")).decode("utf-8")
        checksum_src = (api_key + cur_time + param_b64).encode("utf-8")
        checksum = hashlib.md5(checksum_src).hexdigest()
        return {
            "X-Appid": appid,
            "X-CurTime": cur_time,
            "X-Param": param_b64,
            "X-CheckSum": checksum,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "User-Agent": get_user_agent(),
        }

    def _judge_json(self, http_code: int, body: str) -> CheckResult:
        # Treat auth success (even with business error like missing audio) as valid.
        # Strictly fail only on known auth error codes.
        if http_code >= 500:
            return CheckResult.fail(ErrorReason.SERVER_ERROR)
        try:
            data = json.loads(body or "{}")
            code = str(data.get("code", ""))
            desc = str(data.get("desc", ""))
            # Known auth failure codes (examples from docs/blogs)
            auth_fail = {"10105", "10107", "10900", "11200", "11201"}
            if code in auth_fail:
                return CheckResult.fail(ErrorReason.INVALID_KEY)
            # Explicit success
            if code == "0":
                return CheckResult.success()
            # Business errors that imply request reached biz layer (e.g. missing/empty audio)
            dlow = desc.lower()
            if any(w in dlow for w in ["audio", "file", "pcm", "data", "param"]) and any(w in dlow for w in ["empty", "missing", "null", "format", "invalid", "error"]):
                return CheckResult.success()

            # Otherwise unknown
            return CheckResult.fail(ErrorReason.UNKNOWN)
        except Exception:
            return CheckResult.fail(ErrorReason.UNKNOWN)

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        api_key = trim(token)
        appid = trim(model)
        if not api_key or not appid:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        headers = self._build_headers(appid=appid, api_key=api_key)
        # Send minimal form body with empty audio
        body = urllib.parse.urlencode({"audio": ""}).encode("utf-8")
        url = "https://iat-api.xfyun.cn/v2/iat"
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                code = resp.getcode()
                text = resp.read().decode("utf-8", errors="replace")
                return self._judge_json(code, text)
        except urllib.error.HTTPError as e:
            code = e.code
            text = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e.reason)
            return self._judge_json(code, text)
        except Exception as e:
            logger.warning(f"iFlytek check error: {e}")
            return CheckResult.fail(ErrorReason.UNKNOWN)

    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        return []


register_provider("iflytek_speech", IflytekSpeechProvider)

