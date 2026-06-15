#!/usr/bin/env python3

"""
Core Types - Fundamental Data Types and Interfaces

This module defines core types, interfaces, and abstract base classes
used throughout the application for type safety and clear contracts.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Protocol

from .metrics import PipelineStatus
from .models import CheckResult, Condition, Patterns, ResultStorage


class IAuthProvider(Protocol):
    """Protocol for resource management with dependency injection support.

    This interface defines the contract for resource providers that can be
    injected into components that need access to credentials and user agents.
    """

    def get_session(self) -> Optional[str]:
        """Get next GitHub session token."""
        ...

    def get_token(self) -> Optional[str]:
        """Get next GitHub API token."""
        ...

    def get_credential(self, prefer_token: bool = True) -> tuple[str, str]:
        """Get next credential with type preference."""
        ...

    def get_user_agent(self) -> str:
        """Get random User-Agent string."""
        ...


class IProvider(ABC):
    """Core provider interface with essential properties and methods

    This interface defines the contract that all AI service providers must implement.
    It provides type safety without creating circular dependencies.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier"""
        pass

    @property
    @abstractmethod
    def conditions(self) -> List[Condition]:
        """Search conditions for this provider"""
        pass

    @property
    @abstractmethod
    def result(self) -> ResultStorage:
        """Result storage metadata for this provider"""
        pass

    @abstractmethod
    def get_patterns(self) -> Patterns:
        """Get patterns configuration for this provider

        Returns:
            Patterns: Pattern configuration for key extraction
        """
        pass

    @abstractmethod
    def check(self, token: str, address: str = "", endpoint: str = "", model: str = "", **kwargs) -> CheckResult:
        """Check if token is valid for this provider

        Args:
            token: API token to validate
            address: Optional custom address
            endpoint: Optional custom endpoint
            model: Optional model to test
            kwargs: Optional additional keyword arguments

        Returns:
            CheckResult: Result of the validation check
        """
        pass

    @abstractmethod
    def inspect(self, token: str, address: str = "", endpoint: str = "", **kwargs) -> List[str]:
        """Inspect available models for this provider

        Args:
            token: API token for authentication
            address: Optional custom address
            endpoint: Optional custom endpoint
            kwargs: Optional additional keyword arguments

        Returns:
            List[str]: Available model names
        """
        pass


class IPipelineStats(Protocol):
    """Protocol for pipeline statistics providers"""

    def get_all_stats(self) -> "PipelineStatus":
        """Get comprehensive pipeline statistics

        Returns:
            PipelineStatus: Complete pipeline statistics including all stages
        """
        ...

    def get_dynamic_stats(self) -> "PipelineStatus":
        """Get dynamic pipeline statistics

        Returns:
            PipelineStatus: Current dynamic statistics for active stages
        """
        ...
