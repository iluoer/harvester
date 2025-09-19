#!/usr/bin/env python3

"""
Core Enums - Fundamental Enumeration Types

This module defines core enumeration types used throughout the application.
These enums provide type safety and clear value definitions for system states,
error reasons, and other categorical data.
"""

from enum import Enum, unique


@unique
class SystemState(Enum):
    """System operational state

    Defines the possible states of the system during its lifecycle.
    Used across all modules for consistent state representation.
    """

    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

    def is_active(self) -> bool:
        """Check if system is in an active state"""
        return self in (SystemState.STARTING, SystemState.RUNNING)

    def is_terminal(self) -> bool:
        """Check if system is in a terminal state"""
        return self in (SystemState.STOPPED, SystemState.ERROR)

    def can_transition_to(self, target_state: "SystemState") -> bool:
        """Check if transition to target state is valid"""
        valid_transitions = {
            SystemState.UNKNOWN: {SystemState.STARTING, SystemState.ERROR},
            SystemState.STARTING: {SystemState.RUNNING, SystemState.ERROR, SystemState.STOPPING},
            SystemState.RUNNING: {SystemState.STOPPING, SystemState.ERROR},
            SystemState.STOPPING: {SystemState.STOPPED, SystemState.ERROR},
            SystemState.STOPPED: {SystemState.STARTING},
            SystemState.ERROR: {SystemState.STARTING, SystemState.STOPPED},
        }
        return target_state in valid_transitions.get(self, set())


@unique
class ErrorReason(Enum):
    """Error reasons for API validation and processing

    Standardized error reasons used across the system for
    consistent error handling and reporting.
    """

    UNKNOWN = "unknown"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    INVALID_TOKEN = "invalid_token"
    INVALID_KEY = "invalid_key"  # Alias for invalid_token
    INSUFFICIENT_QUOTA = "insufficient_quota"
    NO_QUOTA = "no_quota"
    NO_MODEL = "no_model"
    NO_ACCESS = "no_access"  # Access denied or permission issues
    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    SERVICE_UNAVAILABLE = "service_unavailable"

    def is_retryable(self) -> bool:
        """Check if error is retryable"""
        retryable_errors = {
            ErrorReason.NETWORK_ERROR,
            ErrorReason.TIMEOUT,
            ErrorReason.RATE_LIMITED,
            ErrorReason.SERVER_ERROR,
            ErrorReason.SERVICE_UNAVAILABLE,
        }
        return self in retryable_errors

    def is_client_error(self) -> bool:
        """Check if error is a client-side error"""
        client_errors = {
            ErrorReason.INVALID_TOKEN,
            ErrorReason.BAD_REQUEST,
            ErrorReason.UNAUTHORIZED,
            ErrorReason.FORBIDDEN,
            ErrorReason.NOT_FOUND,
        }
        return self in client_errors


@unique
class LoadBalanceStrategy(Enum):
    """Load balancing strategy enumeration

    Defines the available strategies for distributing load
    across multiple resources in the system.
    """

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    HEALTHY_FIRST = "healthy_first"

    def get_display_name(self) -> str:
        """Get human-readable display name"""
        display_names = {
            LoadBalanceStrategy.ROUND_ROBIN: "Round Robin",
            LoadBalanceStrategy.RANDOM: "Random",
            LoadBalanceStrategy.HEALTHY_FIRST: "Healthy First",
        }
        return display_names.get(self, self.value.replace('_', ' ').title())


@unique
class ResultType(Enum):
    """Types of results that can be stored

    Defines the different types of results that the system
    can collect and persist during processing.
    """

    VALID = "valid"
    NO_QUOTA = "no_quota"
    WAIT_CHECK = "wait_check"
    INVALID = "invalid"
    MATERIAL = "material"
    LINKS = "links"
    MODELS = "models"
    SUMMARY = "summary"
    INSPECT = "inspect"


@unique
class PipelineStage(Enum):
    """Pipeline stage names for type safety"""

    SEARCH = "search"
    GATHER = "gather"
    CHECK = "check"
    INSPECT = "inspect"


@unique
class QueueStateProvider(Enum):
    """Provider type for queue state management"""

    SINGLE = "single"
    MULTI = "multi"


@unique
class QueueStateStatus(Enum):
    """Status of queue state for monitoring and management"""

    ACTIVE = "active"
    EMPTY = "empty"
    ERROR = "error"
    ARCHIVED = "archived"
    STALE = "stale"
    UNKNOWN = "unknown"


@unique
class QueueOperation(Enum):
    """Queue operation types for logging and monitoring"""

    SAVE = "save"
    LOAD = "load"
    CLEAR = "clear"
    ARCHIVE = "archive"
    FLUSH = "flush"


@unique
class QueueStateField(Enum):
    """Field names for queue state serialization"""

    STAGE = "stage"
    PROVIDER = "provider"
    TASK_COUNT = "task_count"
    SAVED_AT = "saved_at"
    TASKS = "tasks"
    STATUS = "status"
    AGE_HOURS = "age_hours"
    FILE_SIZE = "file_size"
    ERROR = "error"


@unique
class AlertKeyType(Enum):
    """Alert key types for alert deduplication"""

    PERFORMANCE = "performance"
    SYSTEM = "system"
    PROVIDER = "provider"
    PIPELINE = "pipeline"
    RESOURCE = "resource"


@unique
class ErrorType(Enum):
    """Error types for structured error handling"""

    DATA_COLLECTION_ERROR = "DATA_COLLECTION_ERROR"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
