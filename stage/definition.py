#!/usr/bin/env python3

"""
Built-in stage definitions for the pipeline system.
Registers all standard pipeline stages with their dependencies.
"""

import math
import threading
import time
from typing import List, Optional, Tuple

from constant.search import (
    API_LIMIT,
    API_MAX_PAGES,
    API_RESULTS_PER_PAGE,
    WEB_LIMIT,
    WEB_MAX_PAGES,
    WEB_RESULTS_PER_PAGE,
)
from constant.system import SERVICE_TYPE_GITHUB_API, SERVICE_TYPE_GITHUB_WEB
from core.enums import ErrorReason, PipelineStage, ResultType
from core.models import (
    AcquisitionTask,
    CheckTask,
    InspectTask,
    Patterns,
    ProviderTask,
    SearchTask,
    Service,
)
from core.types import IProvider
from refine.engine import RefineEngine
from search import client
from tools.logger import get_logger
from tools.utils import get_service_name, handle_exceptions

from .base import BasePipelineStage, OutputHandler, StageOutput, StageResources
from .factory import TaskFactory
from .registry import register_stage

logger = get_logger("stage")


@register_stage(
    name=PipelineStage.SEARCH.value,
    depends_on=[],
    produces_for=[PipelineStage.GATHER.value, PipelineStage.CHECK.value],
    description="Search GitHub for potential API keys",
)
class SearchStage(BasePipelineStage):
    """Pipeline stage for searching GitHub with pure functional processing"""

    def __init__(self, resources: StageResources, handler: OutputHandler, **kwargs):
        super().__init__(PipelineStage.SEARCH.value, resources, handler, **kwargs)

    def _generate_id(self, task: ProviderTask) -> str:
        """Generate unique task identifier for deduplication"""
        search_task = task if isinstance(task, SearchTask) else SearchTask()
        return (
            f"{PipelineStage.SEARCH.value}:{task.provider}:{search_task.query}:{search_task.page}:{search_task.regex}"
        )

    def _validate_task_type(self, task: ProviderTask) -> bool:
        """Validate that task is a SearchTask."""
        return isinstance(task, SearchTask)

    def _pre_process(self, task: ProviderTask) -> bool:
        """Pre-process search task - validate query and provider."""

        # Check if provider is enabled
        if not self.resources.is_enabled(task.provider, "search"):
            logger.debug(f"[{self.name}] search disabled for provider: {task.provider}")
            return False

        # Validate query
        search_task = task if isinstance(task, SearchTask) else None
        if not search_task or not search_task.query:
            logger.warning(f"[{self.name}] empty query for provider: {task.provider}")
            return False

        return True

    def _execute_task(self, task: ProviderTask) -> Optional[StageOutput]:
        """Execute search task processing."""
        return self._search_worker(task)

    def _search_worker(self, task: SearchTask) -> Optional[StageOutput]:
        """Pure functional search worker"""
        try:
            # Execute search based on page number
            if task.page == 1:
                results, content, total = self._execute_first_page_search(task)
            else:
                results, content = self._execute_page_search(task)
                total = 0

            # Optional fallback: if API returned nothing OR error, try web session once (guarded by extras.fallback_web)
            try:
                should_fallback = (
                    task.page == 1 and task.use_api and (not results) and (total == 0 or not content)
                )
                # Additionally fallback if client layer marked rate/abuse (401/403/429) on first page
                # We piggyback on last log or a soft flag in content; since we don't pass error codes here,
                # we conservatively fallback when content is empty and results empty.
                if should_fallback:
                    task_cfg = self.resources.task_configs.get(task.provider)
                    fallback_enabled = bool(getattr(task_cfg, "extras", {}).get("fallback_web", False)) if task_cfg else False
                    # Only fallback if we have a session available
                    if fallback_enabled and self.resources.auth.get_session() is not None:
                        logger.info(f"[{self.name}] API returned 0/empty for {task.provider}, falling back to web session once for query: {task.query}")
                        # Execute a one-off web search for first page
                        # Note: search_with_count returns (results, total, content)
                        w_results, w_total, w_content = client.search_with_count(
                            query=self._preprocess_query(task.query, False),
                            session=self.resources.auth.get_session(),
                            page=1,
                            with_api=False,
                            peer_page=WEB_RESULTS_PER_PAGE,
                        )
                        # If web found anything, override current variables
                        if w_results or (isinstance(w_total, (int, float)) and w_total > 0):
                            results, content, total = w_results, w_content, w_total
            except Exception:
                pass

            # Create output object
            output = StageOutput(task=task)

            # Extract keys directly from search content
            keys = []
            if content and task.regex:
                keys = self._extract_keys_from_content(content, task)
                scheduled = 0
                skipped_half = 0
                for key_service in keys:
                    # If endpoint is required (endpoint_pattern configured) but missing, skip immediate check
                    ep_required = bool(task.endpoint_pattern)
                    has_endpoint = bool(getattr(key_service, 'endpoint', '') or '')
                    pair_half = ''
                    try:
                        pair_half = (getattr(key_service, 'meta', {}) or {}).get('pair_half', '')
                    except Exception:
                        pair_half = ''
                    if ep_required and (not has_endpoint or pair_half):
                        skipped_half += 1
                        continue
                    check_task = TaskFactory.create_check_task(task.provider, key_service)
                    output.add_task(check_task, PipelineStage.CHECK.value)
                    scheduled += 1

                if keys:
                    logger.info(
                        f"[{self.name}] extracted {len(keys)} keys from search content, scheduled checks: {scheduled}, skipped half: {skipped_half}, provider: {task.provider}"
                    )

            # Create acquisition tasks for links
            if results:
                patterns = Patterns(
                    key_pattern=task.regex,
                    address_pattern=task.address_pattern,
                    endpoint_pattern=task.endpoint_pattern,
                    model_pattern=task.model_pattern,
                )
                for link in results:
                    acquisition_task = TaskFactory.create_acquisition_task(task.provider, link, patterns)
                    output.add_task(acquisition_task, PipelineStage.GATHER.value)

                # Add links to be saved
                output.add_links(task.provider, results)

            # Handle first page results for pagination/refinement
            if task.page == 1 and total > 0:
                self._handle_first_page_results(task, total, output)

            logger.info(
                f"[{self.name}] search completed for {task.provider}: {len(results) if results else 0} links, {len(keys)} keys"
            )

            return output

        except Exception as e:
            logger.error(f"[{self.name}] error, provider: {task.provider}, task: {task}, message: {e}")
            return None

    def _execute_first_page_search(self, task: SearchTask) -> Tuple[List[str], str, int]:
        """Execute first page search and get total count in single request"""
        # Apply rate limiting
        self._apply_rate_limit(task.use_api)

        # Get auth via injected provider
        if task.use_api:
            auth_token = self.resources.auth.get_token()
            # If token is on cooldown, skip this attempt early
            if client.is_token_on_cooldown(auth_token):
                return [], "", 0
        else:
            auth_token = self.resources.auth.get_session()
            if client.is_session_on_cooldown(auth_token):
                return [], "", 0

        # Execute search with count - now returns content as well
        # search_with_count returns (results, total, content)
        results, total, content = client.search_with_count(
            query=self._preprocess_query(task.query, task.use_api),
            session=auth_token,
            page=task.page,
            with_api=task.use_api,
            peer_page=API_RESULTS_PER_PAGE if task.use_api else WEB_RESULTS_PER_PAGE,
        )

        return results, content, total

    def _preprocess_query(self, query: str, use_api: bool) -> str:
        """Github Rest API search syntax don't support regex, so we need remove it if exists"""
        if use_api:
            # Only apply RefineEngine.clean_regex if the query contains regex patterns (starts and ends with /)
            # For simple keyword queries, use them as-is to avoid over-processing
            if query.strip().startswith('/') and query.strip().endswith('/'):
                keyword = RefineEngine.get_instance().clean_regex(query=query)
                if keyword:
                    query = keyword
            # For simple keyword queries like "AKID SecretKey", use them directly

        return query

    def _execute_page_search(self, task: SearchTask) -> Tuple[List[str], str]:
        """Execute subsequent page search in single request"""
        # Apply rate limiting
        self._apply_rate_limit(task.use_api)

        # Get auth via injected provider
        if task.use_api:
            auth_token = self.resources.auth.get_token()
            if client.is_token_on_cooldown(auth_token):
                return [], ""
        else:
            auth_token = self.resources.auth.get_session()
            if client.is_session_on_cooldown(auth_token):
                return [], ""

        # Execute search - now returns content as well
        results, content = client.search_code(
            query=self._preprocess_query(task.query, task.use_api),
            session=auth_token,
            page=task.page,
            with_api=task.use_api,
            peer_page=API_RESULTS_PER_PAGE if task.use_api else WEB_RESULTS_PER_PAGE,
        )

        return results, content

    def _apply_rate_limit(self, use_api: bool) -> bool:
        """Apply rate limiting for GitHub requests"""
        service_type = SERVICE_TYPE_GITHUB_API if use_api else SERVICE_TYPE_GITHUB_WEB
        if not self.resources.limiter.acquire(service_type):
            wait_time = self.resources.limiter.wait_time(service_type)
            if wait_time > 0:
                time.sleep(wait_time)
                if not self.resources.limiter.acquire(service_type):
                    bucket = self.resources.limiter._get_bucket(service_type)
                    max_value = bucket.burst if bucket else "unknown"
                    logger.info(
                        f'[{self.name}] rate limit exceeded for Github {"Rest API" if use_api else "Web"}, max: {max_value}'
                    )
                    return False
        return True

    def _handle_first_page_results(self, task: SearchTask, total: int, output: StageOutput) -> None:
        """Handle first page results - decide pagination or refinement"""
        limit = API_LIMIT if task.use_api else WEB_LIMIT
        per_page = API_RESULTS_PER_PAGE if task.use_api else WEB_RESULTS_PER_PAGE

        # Check per-provider extras to optionally disable refine
        refine_enabled = True
        try:
            task_cfg = self.resources.task_configs.get(task.provider)
            if task_cfg and getattr(task_cfg, 'extras', None):
                refine_enabled = bool(task_cfg.extras.get('refine', True))
        except Exception:
            refine_enabled = True

        # If needs refine query and refine is enabled
        if total > limit and refine_enabled:
            # Regenerate the query with less data
            partitions = int(math.ceil(total / limit))
            queries = RefineEngine.get_instance().generate_queries(query=task.query, partitions=partitions)

            # Cap refine fan-out to avoid abuse detection
            queries = queries[:10]

            # Add new query tasks to output
            for query in queries:
                if not query:
                    logger.warning(
                        f"[{self.name}] skip refined query due to empty for query: {task.query}, provider: {task.provider}"
                    )
                    continue
                elif query == task.query:
                    logger.warning(
                        f"[{self.name}] discard refined query same as original: {query}, provider: {task.provider}"
                    )
                    continue

                refined_task = SearchTask(
                    provider=task.provider,
                    query=query,
                    regex=task.regex,
                    page=1,
                    use_api=task.use_api,
                    address_pattern=task.address_pattern,
                    endpoint_pattern=task.endpoint_pattern,
                    model_pattern=task.model_pattern,
                )

                output.add_task(refined_task, PipelineStage.SEARCH.value)

            logger.info(
                f"[{self.name}] generated {len(queries)} refined tasks (capped) for provider: {task.provider}, query: {task.query}"
            )

        # If needs pagination and not refining
        elif total > per_page:
            page_tasks = self._generate_page_tasks(task, total, per_page)
            for page_task in page_tasks:
                output.add_task(page_task, PipelineStage.SEARCH.value)
            logger.info(
                f"[{self.name}] generated {len(page_tasks)} page tasks for provider: {task.provider}, query: {task.query}"
            )
        else:
            # Fallback: when total is unknown or <= per_page (e.g., API 401/403 blocking count),
            # proactively fan out a minimal set of pagination tasks if configured.
            try:
                task_cfg = self.resources.task_configs.get(task.provider)
                extras = getattr(task_cfg, 'extras', {}) if task_cfg else {}
                min_pages = int(extras.get('min_pages', 0) or 0)
            except Exception:
                min_pages = 0
            if min_pages and min_pages > 1:
                forced_tasks: List[SearchTask] = []
                for page in range(2, min_pages + 1):
                    forced_tasks.append(
                        SearchTask(
                            provider=task.provider,
                            query=task.query,
                            regex=task.regex,
                            page=page,
                            use_api=task.use_api,
                            address_pattern=task.address_pattern,
                            endpoint_pattern=task.endpoint_pattern,
                            model_pattern=task.model_pattern,
                        )
                    )
                for page_task in forced_tasks:
                    output.add_task(page_task, PipelineStage.SEARCH.value)
                logger.info(
                    f"[{self.name}] generated {len(forced_tasks)} forced page tasks (min_pages) for provider: {task.provider}, query: {task.query}"
                )

    def _generate_page_tasks(self, task: SearchTask, total: int, per_page: int) -> List[SearchTask]:
        """Generate pagination tasks"""
        # Limit max pages
        max_pages = min(
            math.ceil(total / per_page),
            API_MAX_PAGES if task.use_api else WEB_MAX_PAGES,
        )

        page_tasks: List[SearchTask] = []
        for page in range(2, max_pages + 1):  # Start from page 2
            page_task = SearchTask(
                provider=task.provider,
                query=task.query,
                regex=task.regex,
                page=page,
                use_api=task.use_api,
                address_pattern=task.address_pattern,
                endpoint_pattern=task.endpoint_pattern,
                model_pattern=task.model_pattern,
            )
            page_tasks.append(page_task)

        return page_tasks

    @handle_exceptions(default_result=[], log_level="error")
    def _extract_keys_from_content(self, content: str, task: SearchTask) -> List[Service]:
        """Extract keys directly from search content"""
        services = client.collect(
            key_pattern=task.regex,
            address_pattern=task.address_pattern,
            endpoint_pattern=task.endpoint_pattern,
            model_pattern=task.model_pattern,
            text=content,
        )

        # Ensure provider metadata is present for cross-file pairing logic
        try:
            for svc in services or []:
                if getattr(svc, 'meta', None) is None:
                    svc.meta = {}
                if 'provider' not in svc.meta:
                    svc.meta['provider'] = (task.provider or '')
        except Exception:
            pass

        return services


@register_stage(
    name=PipelineStage.GATHER.value,
    depends_on=[PipelineStage.SEARCH.value],
    produces_for=[PipelineStage.CHECK.value],
    description="Gather keys from discovered URLs",
)

class AcquisitionStage(BasePipelineStage):
    """Pipeline stage for acquiring keys from URLs with pure functional processing"""

    # --- RepoPairIndex: shared across AcquisitionStage instances ---
    _repo_pair_index = {}
    _repo_pair_lock = threading.Lock()

    @staticmethod
    def _repo_key(meta: dict) -> Optional[tuple]:
        try:
            owner = (meta or {}).get("repo_owner")
            repo = (meta or {}).get("repo_name")
            if owner and repo:
                return (owner, repo)
            return None
        except Exception:
            return None

    @classmethod
    def _try_instant_pair(cls, svc: Service) -> Optional[Service]:
        try:
            meta = getattr(svc, "meta", {}) or {}
            repo_key = cls._repo_key(meta)
            if not repo_key:
                return None
            provider = (meta.get("provider") or "").lower()
            with cls._repo_pair_lock:
                bucket = cls._repo_pair_index.setdefault(repo_key, {"key_only": set(), "endpoint_only": set()})
                half = meta.get("pair_half")
                if half == "key_only":
                    # We have key; try to find the "other half" (endpoint/appid)
                    if bucket["endpoint_only"]:
                        other = next(iter(bucket["endpoint_only"]))
                        bucket["endpoint_only"].discard(other)
                        return Service(address=svc.address, endpoint=other, key=svc.key, model=svc.model or other, meta=meta)
                    bucket["key_only"].add(svc.key)
                    return None
                elif half == "endpoint_only":
                    # We have endpoint/appid; try to find key
                    if bucket["key_only"]:
                        k_val = next(iter(bucket["key_only"]))
                        bucket["key_only"].discard(k_val)
                        return Service(address=svc.address, endpoint=svc.endpoint, key=k_val, model=svc.model or svc.endpoint, meta=meta)
                    # Save the "other half" value for later pairing
                    other_val = svc.endpoint or svc.model
                    if other_val:
                        bucket["endpoint_only"].add(other_val)
                    return None
                else:
                    return None
        except Exception:
            return None

    def __init__(self, resources: StageResources, handler: OutputHandler, **kwargs):
        super().__init__(PipelineStage.GATHER.value, resources, handler, **kwargs)

    def _generate_id(self, task: ProviderTask) -> str:
        """Generate unique task identifier for deduplication"""
        acquisition_task = task if isinstance(task, AcquisitionTask) else AcquisitionTask()
        return f"{PipelineStage.GATHER.value}:{task.provider}:{acquisition_task.url}"

    def _validate_task_type(self, task: ProviderTask) -> bool:
        """Validate that task is an AcquisitionTask."""
        return isinstance(task, AcquisitionTask)

    def _execute_task(self, task: ProviderTask) -> Optional[StageOutput]:
        """Execute acquisition task processing."""
        return self._acquisition_worker(task)

    def _acquisition_worker(self, task: AcquisitionTask) -> Optional[StageOutput]:
        """Pure functional acquisition worker implementation"""
        try:
            # Execute acquisition using global collect function
            services = client.collect(
                key_pattern=task.key_pattern,
                url=task.url,
                retries=task.retries,
                address_pattern=task.address_pattern,
                endpoint_pattern=task.endpoint_pattern,
                model_pattern=task.model_pattern,
            )

            # Stamp provider name into meta for pairing logic downstream
            try:
                for svc in services or []:
                    if getattr(svc, 'meta', None) is None:
                        svc.meta = {}
                    if 'provider' not in svc.meta:
                        svc.meta['provider'] = (task.provider or '')
            except Exception:
                pass

            # Create output object
            output = StageOutput(task=task)

            # Determine provider extras
            extras = {}
            try:
                cfg = self.resources.task_configs.get(task.provider)
                if cfg and getattr(cfg, 'extras', None):
                    extras = cfg.extras
            except Exception:
                extras = {}

            # Helper: produce repo-scoped search tasks to complete pairs
            def _emit_repo_pair_searches(svc: Service) -> None:
                if not isinstance(svc, Service):
                    return
                meta = getattr(svc, 'meta', {}) or {}
                owner = meta.get('repo_owner')
                repo = meta.get('repo_name')
                if not owner or not repo:
                    return
                pair_half = meta.get('pair_half')
                if not pair_half:
                    return
                # Cap queries per half-candidate
                queries: List[str] = []
                repo_scope = f"repo:{owner}/{repo}"
                prov = (task.provider or '').lower()
                if prov == 'tencent_asr':
                    if pair_half == 'key_only':
                        queries = [f"{repo_scope} AKID", f"{repo_scope} SecretId"]
                    elif pair_half == 'endpoint_only':
                        queries = [f"{repo_scope} SecretKey", f"{repo_scope} TENCENTCLOUD_SECRETKEY"]
                elif prov == 'doubao':
                    if pair_half == 'key_only':
                        # have key, need endpointId (ep-...)
                        queries = [
                            f"{repo_scope} ep-",
                            f"{repo_scope} endpointId",
                            f"{repo_scope} ep- path:.github/workflows",
                        ]
                    elif pair_half == 'endpoint_only':
                        # have endpointId, need key/token
                        queries = [
                            f"{repo_scope} ARK_API_KEY filename:.env",
                            f"{repo_scope} VOLCENGINE_ACCESS_TOKEN filename:.env",
                            f"{repo_scope} VOLC_ACCESS_TOKEN filename:.env",
                            f"{repo_scope} ARK_API_KEY path:.github/workflows",
                        ]
                # qianfan provider removed
                if not queries:
                    return
                # Bound the number of follow-up searches
                queries = queries[: extras.get('repo_pair_search_cap', 2) ]
                for q in queries:
                    # Choose regex according to which half we are missing
                    desired_regex = task.key_pattern or ''
                    try:
                        if prov in ('tencent_asr', 'doubao'):
                            if pair_half == 'key_only':
                                # search endpoints/appid for these providers
                                desired_regex = task.endpoint_pattern or ''
                            elif pair_half == 'endpoint_only':
                                desired_regex = task.key_pattern or ''
                    except Exception:
                        desired_regex = task.key_pattern or ''

                    st = TaskFactory.create_search_task(
                        provider=task.provider,
                        query=q,
                        regex=desired_regex,
                        page=1,
                        use_api=True,
                        address_pattern=task.address_pattern or '',
                        endpoint_pattern=task.endpoint_pattern or '',
                        model_pattern=task.model_pattern or '',
                    )
                    output.add_task(st, PipelineStage.SEARCH.value)

            # Create check tasks for found services; defer half-candidates to repo pairing
            if services:
                complete: List[Service] = []
                for service in services:
                    half = False
                    try:
                        meta = getattr(service, 'meta', {}) or {}
                        if meta.get('pair_half'):
                            half = True
                    except Exception:
                        half = False
                    if half and extras.get('cross_pair', True):
                        # First try instant cross-file pairing via RepoPairIndex
                        paired = AcquisitionStage._try_instant_pair(service)
                        if paired:
                            complete.append(paired)
                        else:
                            _emit_repo_pair_searches(service)
                        # Do not send original half-candidate to check yet
                        continue
                    complete.append(service)

                for svc in complete:
                    check_task = TaskFactory.create_check_task(task.provider, svc)
                    output.add_task(check_task, PipelineStage.CHECK.value)

                # Add material keys to be saved (include both complete and half-candidates for traceability)
                output.add_result(task.provider, ResultType.MATERIAL.value, services)

            # Add the processed link to be saved
            output.add_links(task.provider, [task.url])

            return output

        except Exception as e:
            logger.error(f"[{self.name}] error for provider: {task.provider}, task: {task}, message: {e}")
            return None


@register_stage(
    name=PipelineStage.CHECK.value,
    depends_on=[],
    produces_for=[PipelineStage.INSPECT.value],
    description="Validate API keys",
)
class CheckStage(BasePipelineStage):
    """Pipeline stage for validating API keys with pure functional processing"""

    def __init__(self, resources: StageResources, handler: OutputHandler, **kwargs):
        super().__init__(PipelineStage.CHECK.value, resources, handler, **kwargs)

    def _generate_id(self, task: ProviderTask) -> str:
        """Generate unique task identifier for deduplication"""
        check_task = task if isinstance(task, CheckTask) else None
        if check_task and check_task.service:
            service = check_task.service
            return f"{PipelineStage.CHECK.value}:{task.provider}:{service.key}:{service.address}:{service.endpoint}"

        return f"{PipelineStage.CHECK.value}:{task.provider}:unknown"

    def _validate_task_type(self, task: ProviderTask) -> bool:
        """Validate that task is a CheckTask."""
        return isinstance(task, CheckTask)

    def _execute_task(self, task: ProviderTask) -> Optional[StageOutput]:
        """Execute check task processing."""
        return self._check_worker(task)

    def _check_worker(self, task: CheckTask) -> Optional[StageOutput]:
        """Pure functional check worker implementation"""
        try:
            # Get provider instance
            provider = self.resources.providers.get(task.provider)
            if not provider or not isinstance(provider, IProvider):
                logger.error(f"[{self.name}] unknown provider: {task.provider}, type: {type(provider)}")
                return None

            # Apply rate limiting
            service_type = get_service_name(task.provider)
            if not self.resources.limiter.acquire(service_type):
                wait_time = self.resources.limiter.wait_time(service_type)
                if wait_time > 0:
                    time.sleep(wait_time)
                    if not self.resources.limiter.acquire(service_type):
                        bucket = self.resources.limiter._get_bucket(service_type)
                        max_value = bucket.burst if bucket else "unknown"
                        logger.info(
                            f"[{self.name}] rate limit exceeded for provider: {task.provider}, max: {max_value}"
                        )
                        return None

            # Execute check
            result = provider.check(
                token=task.service.key,
                address=task.custom_url or task.service.address,
                endpoint=task.service.endpoint,
                model=task.service.model,
            )

            # Report rate limit success
            self.resources.limiter.report_result(service_type, True)

            # Create output object
            output = StageOutput(task=task)

            # Handle result based on availability
            if result.available:
                # Enrich service with optional account info (e.g., balances) before persisting
                try:
                    markers = provider.inspect(
                        token=task.service.key,
                        address=task.custom_url or task.service.address,
                        endpoint=task.service.endpoint,
                    )
                    meta = {}
                    models_list = None
                    for m in markers or []:
                        if isinstance(m, str) and "=" in m:
                            k, v = m.split("=", 1)
                            k = (k or "").strip()
                            v = (v or "").strip()
                            if k in ("balance", "chargeBalance", "totalBalance", "status", "paid"):
                                if k == "paid":
                                    meta[k] = v.lower() in ("true", "1", "yes")
                                else:
                                    meta[k] = v
                            elif k == "models":
                                try:
                                    models_list = [x.strip() for x in v.split(",") if x.strip()]
                                except Exception:
                                    models_list = None
                    if meta or models_list:
                        # attach to service so it is serialized with the valid record
                        try:
                            if meta:
                                task.service.meta.update(meta)
                            if models_list:
                                task.service.meta["models"] = models_list
                        except Exception:
                            # if meta isn't available on older Service versions, ignore
                            pass
                except Exception:
                    # enrichment is best-effort; do not fail the check flow
                    pass

                # Create inspect task
                inspect_task = TaskFactory.create_inspect_task(task.provider, task.service)
                output.add_task(inspect_task, PipelineStage.INSPECT.value)

                # For Tencent, only persist to valid when we have balance info to meet format expectation
                if (task.provider or '').lower() == 'tencent_asr':
                    try:
                        meta = getattr(task.service, 'meta', {}) or {}
                        if 'balance' not in meta:
                            # Without balance, do not persist to valid list (treat as WAIT_CHECK)
                            output.add_result(task.provider, ResultType.WAIT_CHECK.value, [task.service])
                            return output
                    except Exception:
                        output.add_result(task.provider, ResultType.WAIT_CHECK.value, [task.service])
                        return output

                # Add valid key to be saved
                output.add_result(task.provider, ResultType.VALID.value, [task.service])

            else:
                # Categorize based on error reason
                if result.reason == ErrorReason.NO_QUOTA:
                    output.add_result(task.provider, ResultType.NO_QUOTA.value, [task.service])

                elif result.reason in [
                    ErrorReason.RATE_LIMITED,
                    ErrorReason.NO_MODEL,
                    ErrorReason.NO_ACCESS,
                ]:
                    output.add_result(task.provider, ResultType.WAIT_CHECK.value, [task.service])

                else:
                    output.add_result(task.provider, ResultType.INVALID.value, [task.service])

            return output

        except Exception as e:
            # Report rate limit failure
            self.resources.limiter.report_result(get_service_name(task.provider), False)
            logger.error(f"[{self.name}] error for provider: {task.provider}, task: {task}, message: {e}")

            return None


@register_stage(
    name=PipelineStage.INSPECT.value,
    depends_on=[],
    produces_for=[],
    description="Inspect API capabilities for validated keys",
)
class InspectStage(BasePipelineStage):
    """Pipeline stage for inspecting API capabilities with pure functional processing"""

    def __init__(self, resources: StageResources, handler: OutputHandler, **kwargs):
        super().__init__(PipelineStage.INSPECT.value, resources, handler, **kwargs)

    def _generate_id(self, task: ProviderTask) -> str:
        """Generate unique task identifier for deduplication"""
        inspect_task = task if isinstance(task, InspectTask) else None
        if inspect_task and inspect_task.service:
            service = inspect_task.service
            return f"{PipelineStage.INSPECT.value}:{task.provider}:{service.key}:{service.address}"

        return f"{PipelineStage.INSPECT.value}:{task.provider}:unknown"

    def _validate_task_type(self, task: ProviderTask) -> bool:
        """Validate that task is an InspectTask."""
        return isinstance(task, InspectTask)

    def _execute_task(self, task: ProviderTask) -> Optional[StageOutput]:
        """Execute inspect task processing."""
        return self._inspect_worker(task)

    def _inspect_worker(self, task: InspectTask) -> Optional[StageOutput]:
        """Pure functional inspect worker implementation"""
        try:
            # Get provider instance
            provider = self.resources.providers.get(task.provider)
            if not provider or not isinstance(provider, IProvider):
                logger.error(f"[{self.name}] unknown provider: {task.provider}, type: {type(provider)}")
                return None

            # Inspect provider for markers (may include balances, paid flags, or models)
            markers = provider.inspect(
                token=task.service.key, address=task.service.address, endpoint=task.service.endpoint
            )

            # Create output object
            output = StageOutput(task=task)

            # Parse markers into meta/models like in CHECK stage
            meta: Dict[str, Any] = {}
            models_list = None
            try:
                for m in markers or []:
                    if isinstance(m, str) and "=" in m:
                        k, v = m.split("=", 1)
                        k = (k or "").strip()
                        v = (v or "").strip()
                        if k in ("balance", "chargeBalance", "totalBalance", "status", "paid"):
                            if k == "paid":
                                meta[k] = v.lower() in ("true", "1", "yes")
                            else:
                                meta[k] = v
                        elif k == "models":
                            try:
                                models_list = [x.strip() for x in v.split(",") if x.strip()]
                            except Exception:
                                models_list = None
                if meta:
                    try:
                        task.service.meta.update(meta)
                    except Exception:
                        pass
            except Exception:
                pass

            # Save models list if any
            if models_list:
                output.add_models(task.provider, task.service.key, models_list)

            # For Tencent, persist to valid when balance info is available (align with format requirement)
            if (task.provider or '').lower() == 'tencent_asr':
                try:
                    if 'balance' in (task.service.meta or {}):
                        output.add_result(task.provider, ResultType.VALID.value, [task.service])
                except Exception:
                    pass

            return output

        except Exception as e:
            logger.error(f"[{self.name}] inspect models error, provider: {task.provider}, task: {task}, message: {e}")
            return None
