#!/usr/bin/env python3

"""
Optimization strategies for enumeration.
"""

import itertools
import math
from typing import List, Tuple

from tools.logger import get_logger

from .segment import CharClassSegment
from .types import IOptimizationStrategy

logger = get_logger("refine")


class GreedyStrategy(IOptimizationStrategy):
    """Greedy strategy: select highest value segments first."""

    def __init__(self, max_queries: int = 100000000):
        self.max_queries = max_queries

    def select_segments(self, segments: List[CharClassSegment]) -> List[CharClassSegment]:
        """Select segments greedily by value."""
        if not segments:
            return []

        # Sort by value descending
        sorted_segs = sorted(segments, key=lambda s: s.value, reverse=True)

        # Select segments that fit within query limit
        selected = []
        total_queries = 1

        for segment in sorted_segs:
            depth = self.calculate_depth(segment)
            queries = len(segment.charset) ** depth if depth > 0 else 1

            if total_queries * queries <= self.max_queries:
                selected.append(segment)
                total_queries *= queries
            else:
                break

        return selected or [sorted_segs[0]]  # At least select the best one

    def calculate_depth(self, segment: CharClassSegment) -> int:
        """Calculate optimal depth for greedy strategy."""
        charset_size = len(segment.charset)

        if charset_size <= 1:
            return 1
        elif charset_size <= 10:
            return 3
        elif charset_size <= 36:
            return 2
        else:
            return 1

    def evaluate_combination(self, segments: List[CharClassSegment]) -> Tuple[int, float]:
        """Evaluate combination for greedy strategy."""
        total_queries = 1
        total_value = 0.0

        for segment in segments:
            depth = self.calculate_depth(segment)
            queries = len(segment.charset) ** depth if depth > 0 else 1
            total_queries *= queries
            total_value += segment.value

        return total_queries, total_value


class BalancedStrategy(IOptimizationStrategy):
    """Balanced strategy: balance between value and query count."""

    def __init__(self, max_queries: int = 100000000):
        self.max_queries = max_queries

    def select_segments(self, segments: List[CharClassSegment]) -> List[CharClassSegment]:
        """Select segments with balanced approach."""
        if not segments:
            return []

        # Calculate efficiency score (value / log(queries))
        scored_segments = []
        for segment in segments:
            depth = self.calculate_depth(segment)
            queries = len(segment.charset) ** depth if depth > 0 else 1
            value = segment.value

            if queries > 0:
                efficiency = value / math.log(max(queries, 2))
                scored_segments.append((segment, efficiency, queries))

        # Sort by efficiency descending
        scored_segments.sort(key=lambda x: x[1], reverse=True)

        # Select segments that fit within query limit
        selected = []
        total_queries = 1

        for segment, efficiency, queries in scored_segments:
            if total_queries * queries <= self.max_queries:
                selected.append(segment)
                total_queries *= queries
            else:
                break

        return selected or [scored_segments[0][0]] if scored_segments else []

    def calculate_depth(self, segment: CharClassSegment) -> int:
        """Calculate balanced depth."""
        charset_size = len(segment.charset)
        value = segment.value

        # Adjust depth based on value
        base_depth = 2
        if value > 0.8:
            base_depth = 3
        elif value < 0.3:
            base_depth = 1

        # Adjust based on charset size
        if charset_size <= 10:
            return min(base_depth + 1, 4)
        elif charset_size <= 36:
            return base_depth
        else:
            return max(base_depth - 1, 1)

    def evaluate_combination(self, segments: List[CharClassSegment]) -> Tuple[int, float]:
        """Evaluate combination with balanced approach."""
        total_queries = 1
        total_value = 0.0

        for segment in segments:
            depth = self.calculate_depth(segment)
            queries = len(segment.charset) ** depth if depth > 0 else 1
            total_queries *= queries
            total_value += segment.value

        # Apply penalty for too many queries
        if total_queries > self.max_queries / 10:
            total_value *= 0.5

        return total_queries, total_value


class ConservativeStrategy(IOptimizationStrategy):
    """Conservative strategy: minimize query count while maintaining value."""

    def __init__(self, max_queries: int = 100000000):
        self.max_queries = max_queries

    def select_segments(self, segments: List[CharClassSegment]) -> List[CharClassSegment]:
        """Select segments conservatively."""
        if not segments:
            return []

        # Prefer single high-value segment
        best_segment = max(segments, key=lambda s: s.value)
        return [best_segment]

    def calculate_depth(self, segment: CharClassSegment) -> int:
        """Calculate conservative depth."""
        charset_size = len(segment.charset)

        # Always use minimal depth
        if charset_size <= 10:
            return 2
        elif charset_size <= 36:
            return 1
        else:
            return 1

    def evaluate_combination(self, segments: List[CharClassSegment]) -> Tuple[int, float]:
        """Evaluate combination conservatively."""
        if len(segments) > 1:
            # Penalize multi-segment combinations
            return 1, 0.0

        segment = segments[0]
        depth = self.calculate_depth(segment)
        queries = len(segment.charset) ** depth if depth > 0 else 1
        value = segment.value

        return queries, value


class AggressiveStrategy(IOptimizationStrategy):
    """Aggressive strategy: maximize coverage with higher query counts."""

    def __init__(self, max_queries: int = 100000000):
        self.max_queries = max_queries

    def select_segments(self, segments: List[CharClassSegment]) -> List[CharClassSegment]:
        """Select segments aggressively."""
        if not segments:
            return []

        # Try to select multiple segments
        sorted_segs = sorted(segments, key=lambda s: s.value, reverse=True)

        # Try combinations up to 3 segments
        best_combination = []
        best_score = 0.0

        for combo_size in range(1, min(4, len(sorted_segs) + 1)):
            for combo in itertools.combinations(sorted_segs, combo_size):
                queries, value = self.evaluate_combination(list(combo))
                if queries <= self.max_queries and value > best_score:
                    best_combination = list(combo)
                    best_score = value

        return best_combination or [sorted_segs[0]]

    def calculate_depth(self, segment: CharClassSegment) -> int:
        """Calculate aggressive depth."""
        charset_size = len(segment.charset)

        # Use higher depths for better coverage
        if charset_size <= 10:
            return 4
        elif charset_size <= 36:
            return 3
        else:
            return 2

    def evaluate_combination(self, segments: List[CharClassSegment]) -> Tuple[int, float]:
        """Evaluate combination aggressively."""
        total_queries = 1
        total_value = 0.0

        for segment in segments:
            depth = self.calculate_depth(segment)
            queries = len(segment.charset) ** depth if depth > 0 else 1
            total_queries *= queries
            total_value += segment.value

        # Bonus for multi-segment combinations
        if len(segments) > 1:
            total_value *= 1.2

        return total_queries, total_value
