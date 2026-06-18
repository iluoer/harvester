#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interfaces for refine components to support dependency injection and testing.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from .segment import CharClassSegment, EnumerationStrategy, Segment


class IRegexParser(ABC):
    """Interface for regex pattern parsing."""

    @abstractmethod
    def parse(self, pattern: str) -> List[Segment]:
        """Parse regex pattern into segments."""
        pass


class ISplittabilityAnalyzer(ABC):
    """Interface for pattern splittability analysis."""

    @abstractmethod
    def can_split(self, pattern: str, segments: List[Segment], current_depth: int = 0) -> Tuple[bool, str]:
        """Determine if pattern can be split further."""
        pass


class IEnumerationOptimizer(ABC):
    """Interface for enumeration strategy optimization."""

    @abstractmethod
    def optimize(self, segments: List[Segment], partitions: int = 10):
        """Optimize enumeration strategy for given segments."""
        pass


class IOptimizationStrategy(ABC):
    """Interface for optimization strategies."""

    @abstractmethod
    def select_segments(self, segments: List[CharClassSegment]) -> List[CharClassSegment]:
        """Select segments for enumeration."""
        pass

    @abstractmethod
    def calculate_depth(self, segment: CharClassSegment) -> int:
        """Calculate optimal enumeration depth for a segment."""
        pass

    @abstractmethod
    def evaluate_combination(self, segments: List[CharClassSegment]) -> Tuple[int, float]:
        """Evaluate a combination of segments and return (queries, value)."""
        pass


class IQueryGenerator(ABC):
    """Interface for query generation."""

    @abstractmethod
    def generate(self, segments: List[Segment], strategy: EnumerationStrategy, partitions: int = -1) -> List[str]:
        """Generate queries from enumeration strategy."""
        pass


class IEnumerationOptimizer(ABC):
    """Interface for enumeration optimization."""

    @abstractmethod
    def optimize(self, segments: List[Segment]) -> EnumerationStrategy:
        """Find optimal enumeration strategy for segments."""
        pass
