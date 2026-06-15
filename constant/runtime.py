#!/usr/bin/env python3

"""
Runtime Configuration Constants

This module contains constants related to pipeline stages, queue management,
and data processing workflows used during system runtime.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from core.enums import PipelineStage, ResultType

# Queue sizes for each stage
DEFAULT_SEARCH_QUEUE_SIZE: int = 500000
DEFAULT_GATHER_QUEUE_SIZE: int = 1000000
DEFAULT_CHECK_QUEUE_SIZE: int = 5000000
DEFAULT_INSPECT_QUEUE_SIZE: int = 10000000

# Queue and thread defaults using PipelineStage enum
DEFAULT_THREAD_COUNTS: Dict[str, int] = {
    PipelineStage.SEARCH.value: 1,
    PipelineStage.GATHER.value: 8,
    PipelineStage.CHECK.value: 4,
    PipelineStage.INSPECT.value: 2,
}

# Queue state constants
QUEUE_STATE_PROVIDER_MULTI: str = "multi"
QUEUE_STATE_MAX_AGE_HOURS: int = 24


@dataclass
class ResultMapping:
    """Configuration for result type field mapping"""

    # ResultStorage filenames key
    filename: str

    # PersistenceMetrics field name for persistence stats
    stats: Optional[str] = None


# Result type mapping configuration
RESULT_MAPPINGS: Dict[ResultType, ResultMapping] = {
    ResultType.VALID: ResultMapping(filename="valid", stats="valid"),
    ResultType.INVALID: ResultMapping(filename="invalid", stats="invalid"),
    ResultType.NO_QUOTA: ResultMapping(filename="no_quota", stats="no_quota"),
    ResultType.WAIT_CHECK: ResultMapping(filename="wait_check", stats="wait_check"),
    ResultType.MATERIAL: ResultMapping(filename="material", stats="material"),
    ResultType.LINKS: ResultMapping(filename="links", stats="links"),
    ResultType.SUMMARY: ResultMapping(filename="summary", stats=None),
}
