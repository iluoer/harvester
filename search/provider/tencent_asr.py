#!/usr/bin/env python3

"""
Tencent Cloud ASR provider with TC3-HMAC-SHA256 signature validation.

We validate SecretId/SecretKey by making a signed POST request to asr.tencentcloudapi.com.
We intentionally use a benign "test" request; if signature is valid but action/version
are not, Tencent Cloud returns a JSON error with non-auth error codes (e.g. InvalidAction,
InvalidParameter.Version), which we treat as success. Auth failures (InvalidSecretId,
AuthFailure.SignatureFailure) are treated as invalid key.

Field mapping from Service -> check():
- token   => SecretKey
- endpoint=> SecretId
- address => ignored (host fixed)
- model   => optional region (unused here)

Patterns in config.yaml should capture SecretId (to endpoint/model) and SecretKey (to token).
"""

from __future__ import annotations

import datetime
import hashlib
import hmac
import json
from typing import Dict, List, Optional
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


class TencentASRProvider(AIBaseProvider):
    def __init__(self, conditions: List[Condition], **kwargs):
        self.defaults(
            kwargs,
            {
                "name": "tencent_asr",
                "base_url": "https://asr.tencentcloudapi.com",
                "completion_path": "/",
                "model_path": "",
                "default_model": "ap-guangzhou",  # region not strictly required for auth check
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

    # We don't use the default _get_headers/chat flow
    def _get_headers(self, token: str, additional: Optional[Dict] = None) -> Optional[Dict]:
        return None

    def _judge_json(self, http_code: int, body: str) -> CheckResult:
        # HTTP level quick checks
        if http_code == 401:
            return CheckResult.fail(ErrorReason.INVALID_KEY)
        # Do not immediately fail on 403; first parse JSON to detect UnauthorizedOperation
        # Many Tencent APIs return JSON error bodies on auth-permission issues.
        # 429 and 5xx can still be handled early.
        if http_code == 429:
            return CheckResult.fail(ErrorReason.RATE_LIMITED)
        if http_code >= 500:
            return CheckResult.fail(ErrorReason.SERVER_ERROR)

        # Parse JSON to distinguish auth errors vs other errors
        try:
            data = json.loads(body or "{}")
            resp = data.get("Response", {})
            err = resp.get("Error", {})
            code = str(err.get("Code", ""))

            # Treat lack-of-permission as success (credentials valid but unauthorized for this action)
            # e.g. UnauthorizedOperation, UnauthorizedOperation.CamNoAuth, AuthFailure.UnauthorizedOperation.*
            if code == "UnauthorizedOperation" or code.startswith("UnauthorizedOperation") or code.startswith("AuthFailure.UnauthorizedOperation"):
                return CheckResult.success()

            # Explicit auth failures -> invalid (invalid id/signature)
            if code.startswith("InvalidSecretId") or code.startswith("AuthFailure.InvalidSecretId") or code.startswith("AuthFailure.SignatureFailure"):
                return CheckResult.fail(ErrorReason.INVALID_KEY)

            # Time skew / signature expire are transient, not invalid
            if code in ("AuthFailure.SignatureExpire", "AuthFailure.RequestExpired"):
                return CheckResult.fail(ErrorReason.UNKNOWN)

            # Generic AuthFailure without more detail is ambiguous; treat as invalid to be conservative
            if code.startswith("AuthFailure"):
                return CheckResult.fail(ErrorReason.INVALID_KEY)

            # Whitelist of non-auth, post-auth errors that indicate the request reached business logic
            ok_codes = {
                "InvalidParameter",
                "MissingParameter",
                "InvalidParameterValue",
                "UnsupportedOperation",
                "ResourceNotFound",
                "InvalidAction",
                "UnsupportedRegion",
                "LimitExceeded",
                "RequestLimitExceeded",
                "InternalError",
                "UnknownParameter",
                "InvalidRequest",
                "DryRunOperation",
                "OperationDenied",
                # Common non-auth errors seen with valid creds
                "UnsupportedRegion",
                "RequestLimitExceeded",
            }
            # Treat 200/403 with ok_codes or UnauthorizedOperation* as success
            if http_code in (200, 403) and (code in ok_codes or code == "UnauthorizedOperation" or code.startswith("UnauthorizedOperation") or code.startswith("AuthFailure.UnauthorizedOperation")):
                return CheckResult.success()

            # Unknown error code - conservative: not valid
            return CheckResult.fail(ErrorReason.UNKNOWN)
        except Exception:
            return CheckResult.fail(ErrorReason.UNKNOWN)

    def _tc3_signature(self, secret_id: str, secret_key: str, payload: str, action: str, version: str, region: Optional[str] = None, service: str = "asr", host: Optional[str] = None, timestamp_override: Optional[int] = None) -> Dict[str, str]:
        host = host or f"{service}.tencentcloudapi.com"
        algorithm = "TC3-HMAC-SHA256"
        timestamp = int(timestamp_override if timestamp_override is not None else datetime.datetime.utcnow().timestamp())
        date = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

        canonical_uri = "/"
        canonical_querystring = ""
        canonical_headers = f"content-type:application/json\nhost:{host}\n"
        signed_headers = "content-type;host"
        hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        canonical_request = "\n".join([
            "POST",
            canonical_uri,
            canonical_querystring,
            canonical_headers,
            signed_headers,
            hashed_payload,
        ])

        credential_scope = f"{date}/{service}/tc3_request"
        hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        string_to_sign = "\n".join([
            algorithm,
            str(timestamp),
            credential_scope,
            hashed_canonical_request,
        ])

        def _hmac_sha256(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        secret_date = _hmac_sha256(("TC3" + secret_key).encode("utf-8"), date)
        secret_service = _hmac_sha256(secret_date, service)
        secret_signing = _hmac_sha256(secret_service, "tc3_request")
        signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        authorization = (
            f"{algorithm} Credential={secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Host": host,
            "X-TC-Action": action,
            "X-TC-Version": version,
            "X-TC-Timestamp": str(timestamp),
            "User-Agent": get_user_agent(),
        }
        if region:
            headers["X-TC-Region"] = region
        return headers

    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "") -> CheckResult:
        secret_key = trim(token)
        secret_id = trim(endpoint)
        if not secret_id or not secret_key:
            return CheckResult.fail(ErrorReason.BAD_REQUEST)

        # Use an action/version that exists and requires auth; business params can be empty
        action = "DescribeTaskStatus"
        version = "2019-06-14"
        payload_obj: Dict = {}
        payload = json.dumps(payload_obj, separators=(",", ":"))

        # Determine region attempts: prefer provided model, else try a short fallback list
        primary_region = trim(model) or trim(self._default_model or "") or None
        region_attempts: List[Optional[str]] = []
        if primary_region:
            region_attempts = [primary_region]
        else:
            region_attempts = [None, "ap-guangzhou", "ap-shanghai", "ap-beijing"]

        last_result: Optional[CheckResult] = None
        # Try multiple service hosts to maximize acceptance
        service_hosts = [
            ("asr", None),
            ("tts", None),
            ("aiapi", None),
        ]
        for reg in region_attempts:
            for service, custom_host in service_hosts:
                url = f"https://{(custom_host or f'{service}.tencentcloudapi.com')}/"
                headers = self._tc3_signature(secret_id, secret_key, payload, action, version, region=reg, service=service, host=custom_host)
                req = urllib.request.Request(url=url, data=payload.encode("utf-8"), headers=headers, method="POST")

                try:
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        code = resp.getcode()
                        body = resp.read().decode("utf-8", errors="replace")
                        result = self._judge_json(code, body)
                except urllib.error.HTTPError as e:
                    code = e.code
                    body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e.reason)
                    result = self._judge_json(code, body)
                except Exception as e:
                    logger.warning(f"Tencent ASR check error (region={reg or 'none'}): {e}")
                    result = CheckResult.fail(ErrorReason.UNKNOWN)

                # Short-circuit on definite outcomes
                if result.ok:
                    return result
                if result.reason == ErrorReason.INVALID_KEY:
                    # Invalid regardless of region
                    return result
                last_result = result

        return last_result or CheckResult.fail(ErrorReason.UNKNOWN)
    def inspect(self, token: str, address: str = "", endpoint: str = "") -> List[str]:
        """
        Enrich with account info if possible. Returns markers like:
          - paid=true/false
          - balance=Balance:123,RealBalance:0 (key:value comma-joined)
        """
        out: List[str] = []
        secret_key = trim(token)
        secret_id = trim(endpoint)
        if not secret_id or not secret_key:
            return out
        # Try Billing.DescribeAccountBalance, handle server time drift
        action = "DescribeAccountBalance"
        version = "2018-07-09"
        payload = "{}"
        url = "https://billing.tencentcloudapi.com/"
        ts_override: Optional[int] = None
        for attempt in range(2):
            headers = self._tc3_signature(secret_id, secret_key, payload, action, version,
                                          region=None, service="billing", host=None,
                                          timestamp_override=ts_override)
            req = urllib.request.Request(url=url, data=payload.encode("utf-8"), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8", errors="replace")
                    data = json.loads(body or "{}")
                    code = str(((data or {}).get("Response", {}) or {}).get("Error", {}).get("Code", ""))
                    if code == "AuthFailure.SignatureExpire" and attempt == 0:
                        # parse server time from message
                        import re
                        msg = str(((data or {}).get("Response", {}) or {}).get("Error", {}).get("Message", ""))
                        m = re.search(r"(\d{10})", msg)
                        if m:
                            ts_override = int(m.group(1))
                            continue
                except Exception:
                    pass
                return out
            except Exception:
                return out

            try:
                data = json.loads(body or "{}")
                resp = (data or {}).get("Response", {}) or {}
                # Pick primary balance field if present
                fields = ["Balance", "RealBalance", "CashAccountBalance", "Cash", "VoucherPayAmount"]
                primary = None
                for k in fields:
                    if k in resp:
                        try:
                            val = int(resp.get(k) or 0)
                        except Exception:
                            try:
                                val = int(str(resp.get(k)).strip())
                            except Exception:
                                continue
                        if primary is None:
                            primary = val
                if primary is not None:
                    out.append(f"paid={'true' if primary > 0 else 'false'}")
                    out.append(f"balance={primary}")
            except Exception:
                return out
            break
        return out



    def revalidate_file(self, material_path: str, output_path: str) -> int:
        """
        Revalidate candidate pairs from material_path using the provider's strict check()
        and write truly valid entries to output_path as JSONL with fields {endpoint, key}.

        Returns the number of valid entries written.
        """
        valid_count = 0
        seen: set[tuple[str, str]] = set()
        try:
            with open(material_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    endpoint = trim(obj.get("endpoint", "") or obj.get("secret_id", ""))
                    key = trim(obj.get("key", "") or obj.get("secret_key", ""))
                    model = trim(obj.get("model", ""))
                    if not endpoint or not key:
                        continue
                    pair = (endpoint, key)
                    if pair in seen:
                        continue
                    seen.add(pair)

                    res = self.check(token=key, endpoint=endpoint, model=model)
                    if res and res.ok:
                        fout.write(json.dumps({"endpoint": endpoint, "key": key}, ensure_ascii=False) + "\n")
                        valid_count += 1
        except FileNotFoundError:
            logger.warning(f"Material file not found: {material_path}")
        except Exception as e:
            logger.warning(f"Revalidate error: {e}")
        return valid_count


register_provider("tencent_asr", TencentASRProvider)

