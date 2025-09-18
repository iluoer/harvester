#!/usr/bin/env python3

"""
Enumeration strategy optimizer for regex patterns.
"""

import heapq
import itertools
import math
from typing import Dict, List, Tuple

from tools.logger import get_logger

from .segment import (
    CharClassSegment,
    EnumerationStrategy,
    FixedSegment,
    GroupSegment,
    OptionalSegment,
    Segment,
)
from .strategies import (
    AggressiveStrategy,
    BalancedStrategy,
    ConservativeStrategy,
    GreedyStrategy,
)
from .types import IEnumerationOptimizer, IOptimizationStrategy

logger = get_logger("refine")


class EnumerationOptimizer(IEnumerationOptimizer):
    """Optimize enumeration strategy for regex patterns using pluggable strategies."""

    def __init__(self, max_queries: int = 100000000, strategy: IOptimizationStrategy = None):
        self.max_queries = max_queries
        self.strategy = strategy or BalancedStrategy(max_queries)

        # Available strategies for dynamic switching
        self.strategies = {
            "greedy": GreedyStrategy(max_queries),
            "balanced": BalancedStrategy(max_queries),
            "conservative": ConservativeStrategy(max_queries),
            "aggressive": AggressiveStrategy(max_queries),
        }

    def set_strategy(self, strategy_name: str) -> None:
        """Set optimization strategy by name."""
        if strategy_name in self.strategies:
            self.strategy = self.strategies[strategy_name]
            logger.info(f"Switched to {strategy_name} optimization strategy")
        else:
            logger.warning(f"Unknown strategy: {strategy_name}, keeping current strategy")

    def set_custom_strategy(self, strategy: IOptimizationStrategy) -> None:
        """Set custom optimization strategy."""
        self.strategy = strategy
        logger.info(f"Set custom optimization strategy: {strategy.__class__.__name__}")

    def optimize(self, segments: List[Segment]) -> EnumerationStrategy:
        """Find optimal enumeration strategy."""
        if not segments:
            return EnumerationStrategy([], segments, 0.0, 1)

        try:
            # Expand all variants from optional segments
            variants = self._expand_variants(segments)

            best_strategy = None
            best_value = 0.0

            # Find best strategy across all variants
            for variant in variants:
                strategy = self._optimize_variant(variant)
                if strategy.value > best_value:
                    best_strategy = strategy
                    best_value = strategy.value

            # If no strategy found, ensure we at least try to enumerate available segments
            if best_strategy is None or best_strategy.value == 0.0:
                # Find any enumerable segments and create a basic strategy
                def find_vars(segs):
                    vars = []
                    for seg in segs:
                        if isinstance(seg, CharClassSegment):
                            vars.append(seg)
                        elif isinstance(seg, (GroupSegment, OptionalSegment)):
                            vars.extend(find_vars(seg.content))
                    return vars

                vars = find_vars(segments)
                if vars:
                    # Calculate values for all segments
                    for segment in vars:
                        segment.value = self._calculate_value(segment, segments)

                    # Sort by value and select the best one
                    sorted_segs = sorted(vars, key=lambda s: s.value, reverse=True)
                    best_segment = sorted_segs[0]

                    # Create a strategy with the best segment
                    optimal_depth = self._calculate_segment_optimal_depth(best_segment)
                    queries = len(best_segment.charset) ** optimal_depth if optimal_depth > 0 else 1

                    best_strategy = EnumerationStrategy([best_segment], segments, best_segment.value, queries)
                    logger.info(
                        f"Fallback strategy selected: 1 segment, value={best_strategy.value:.3f}, queries={best_strategy.queries}"
                    )

            return best_strategy or EnumerationStrategy([], segments, 0.0, 1)

        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            return EnumerationStrategy([], segments, 0.0, 1)

    def evaluate_strategies_for_partitions(
        self, segments: List[Segment], partitions: int
    ) -> tuple[EnumerationStrategy, bool]:
        """
        Evaluate strategies to find one that generates >= partitions queries.

        Returns:
            tuple: (best_strategy, found_suitable_strategy)
        """
        if not segments or partitions <= 0:
            return EnumerationStrategy([], segments, 0.0, 1), False

        try:
            # Expand all variants from optional segments
            variants = self._expand_variants(segments)

            suitable = []
            all_strats = []

            # Evaluate all possible strategies
            for variant in variants:
                strategies = self._generate_all_strategies(variant)
                all_strats.extend(strategies)

                # Find strategies that meet partition requirement
                for strategy in strategies:
                    if strategy.queries >= partitions:
                        suitable.append(strategy)

            # If we found suitable strategies, return the one with minimum enumeration depth
            if suitable:
                # Prioritize single-segment strategies over multi-segment ones
                single = [s for s in suitable if len(s.segments) == 1]
                if single:
                    best = self._select_strategy_with_min_depth(single, partitions)
                else:
                    best = self._select_strategy_with_min_depth(suitable, partitions)
                return best, True

            # If no suitable strategy found, return the one that generates most queries
            if all_strats:
                # Prioritize single-segment strategies
                single = [s for s in all_strats if len(s.segments) == 1]
                if single:
                    best = max(single, key=lambda s: s.queries)
                else:
                    best = max(all_strats, key=lambda s: s.queries)
                return best, False

            return EnumerationStrategy([], segments, 0.0, 1), False

        except Exception as e:
            logger.warning(f"Strategy evaluation failed: {e}")
            return EnumerationStrategy([], segments, 0.0, 1), False

    def _generate_all_strategies(self, segments: List[Segment]) -> List[EnumerationStrategy]:
        """Generate all possible enumeration strategies for given segments."""

        # Find all variable segments
        def find_vars(segs):
            vars = []
            for seg in segs:
                if isinstance(seg, CharClassSegment):
                    vars.append(seg)
                elif isinstance(seg, (GroupSegment, OptionalSegment)):
                    vars.extend(find_vars(seg.content))
            return vars

        vars = find_vars(segments)

        if not vars:
            return [EnumerationStrategy([], segments, 0.0, 1)]

        # Calculate enumeration values
        for segment in vars:
            segment.value = self._calculate_value(segment, segments)

        strategies = []

        # Generate strategies for different combinations and depths
        # Prioritize single-segment strategies first, then multi-segment
        for combo_size in range(1, min(4, len(vars) + 1)):
            for combo in itertools.combinations(vars, combo_size):
                # Try different enumeration depths
                for depth in range(1, 5):  # depths 1-4
                    strategy = self._create_strategy_with_depth(list(combo), segments, depth)
                    if strategy.queries > 0:
                        strategies.append(strategy)

        return strategies

    def _create_strategy_with_depth(
        self, segments_to_enum: List[CharClassSegment], all_segments: List[Segment], depth: int
    ) -> EnumerationStrategy:
        """Create enumeration strategy with specific depth."""
        total_queries = 1
        total_value = 0.0

        for segment in segments_to_enum:
            charset_size = len(segment.charset)
            if charset_size > 0:
                # Key fix: depth cannot exceed segment's actual enumerable length
                # For \d (max_length=1), max depth is 1
                # For [a-zA-Z0-9]+ (max_length=large), can have larger depth
                effective_depth = min(depth, segment.max_length)

                # If segment's max length is 1 (like \d), can only enumerate 1 layer
                if segment.max_length == 1:
                    segment_queries = charset_size  # Only charset_size possible values
                else:
                    # For variable length segments (like +, *, {n,m}), can enumerate multiple layers
                    segment_queries = charset_size**effective_depth

                total_queries *= segment_queries
                total_value += segment.value

            # Early termination if queries become too large
            if total_queries > self.max_queries * 100:
                break

        return EnumerationStrategy(segments_to_enum, all_segments, total_value, total_queries)

    def _select_strategy_with_min_depth(
        self, strategies: List[EnumerationStrategy], partitions: int
    ) -> EnumerationStrategy:
        """Select strategy with minimum enumeration depth that meets partition requirement."""

        # Calculate effective depth for each strategy
        def calculate_effective_depth(strategy: EnumerationStrategy) -> float:
            if not strategy.segments:
                return 0.0

            total_depth = 0.0
            for segment in strategy.segments:
                charset_size = len(segment.charset)
                if charset_size > 0 and strategy.queries > 0:
                    # Calculate depth from queries = charset_size^depth
                    segment_contribution = strategy.queries ** (1.0 / len(strategy.segments))
                    if segment_contribution > 0:
                        depth = math.log(segment_contribution) / math.log(charset_size)
                        total_depth += depth

            return total_depth / len(strategy.segments) if strategy.segments else 0.0

        # Calculate fixed context length for enumeration segment
        def calc_context_length(strategy: EnumerationStrategy, depth: int) -> int:
            """Calculate the length of fixed context that can be formed with enumeration segment."""
            if not strategy.segments or len(strategy.segments) != 1:
                return 0

            segment = strategy.segments[0]
            segments = strategy.original
            pos = segment.position

            # Calculate preceding fixed content length
            before = 0
            for i in range(pos - 1, -1, -1):
                if isinstance(segments[i], FixedSegment):
                    before += len(segments[i].content)
                elif isinstance(segments[i], OptionalSegment):
                    continue  # Skip optional segments
                else:
                    break  # Stop at variable/group segments

            # Calculate following fixed content length
            # Only if enumeration fully covers current segment
            after = 0
            remaining = segment.min_length - depth
            if remaining <= 0:
                for i in range(pos + 1, len(segments)):
                    if isinstance(segments[i], FixedSegment):
                        after += len(segments[i].content)
                    else:
                        break

            return before + depth + after

        # Calculate total length of segments being enumerated
        def calc_segment_length(strategy: EnumerationStrategy) -> int:
            if not strategy.segments:
                return 0
            return sum(seg.min_length for seg in strategy.segments)

        # Calculate strategy score for selection
        def calc_score(strategy: EnumerationStrategy) -> tuple:
            depth = calculate_effective_depth(strategy)
            length = calc_segment_length(strategy)
            excess = strategy.queries - partitions
            context = calc_context_length(strategy, max(1, int(math.ceil(depth))))

            # Calculate segment value for tie-breaking
            segment_value = 0.0
            if strategy.segments:
                for segment in strategy.segments:
                    segment_value += getattr(segment, "value", 0.0)

            # Find immediate preceding fixed segment length for tie-breaking
            before = 0
            if strategy.segments:
                pos = strategy.segments[0].position
                for i in range(pos - 1, -1, -1):
                    if isinstance(strategy.original[i], FixedSegment):
                        before = len(strategy.original[i].content)
                        break
                    elif isinstance(strategy.original[i], OptionalSegment):
                        continue
                    else:
                        break

            # Prefer higher segment value first, then shorter segments, longer context, etc.
            return (-segment_value, length, -context, -before, depth, excess)

        suitable = [s for s in strategies if s.queries >= partitions]
        if suitable:
            return min(suitable, key=calc_score)

        return strategies[0] if strategies else EnumerationStrategy([], [], 0.0, 1)

    def _expand_variants(self, segments: List[Segment]) -> List[List[Segment]]:
        """Expand all possible variants from optional segments."""
        variants = [[]]

        for segment in segments:
            new_variants = []

            for variant in variants:
                if isinstance(segment, OptionalSegment):
                    # Add both empty and content variants
                    new_variants.append(variant.copy())  # Empty
                    new_variants.append(variant + segment.content)  # Content
                elif isinstance(segment, GroupSegment):
                    # Flatten group content for processing
                    flattened = segment.flatten()
                    new_variants.append(variant + flattened)
                else:
                    # Regular segment
                    new_variants.append(variant + [segment])

            variants = new_variants

            # Check for exponential growth and optimize if needed
            if len(variants) > 1000:
                # Instead of limiting, optimize by selecting most valuable variants
                variants = self._optimize_variants(variants)
                logger.info(f"Optimized {len(new_variants)} variants to {len(variants)} most valuable ones")

        return variants

    def _optimize_variants(self, variants: List[List[Segment]]) -> List[List[Segment]]:
        """Optimize variants by selecting the most valuable ones."""
        if len(variants) <= 1000:
            return variants

        # Calculate target size
        target_size = min(1000, max(100, len(variants) // 2))

        # Use heapq.nlargest for better performance on large datasets
        # Cache segment scores to avoid recalculation
        segment_score_cache: Dict[str, float] = {}

        def score_variant(variant: List[Segment]) -> float:
            return self._score_variant_cached(variant, segment_score_cache)

        # Get top variants without full sorting
        top_scored = heapq.nlargest(target_size, variants, key=score_variant)

        logger.info(f"Selected {len(top_scored)} highest-value variants from {len(variants)} using optimized selection")
        return top_scored

    def _score_variant_cached(self, variant: List[Segment], cache: Dict[str, float]) -> float:
        """Calculate variant score with segment-level caching for better performance"""
        total_score = 0.0

        for segment in variant:
            # Create cache key based on segment type and content
            cache_key = self._get_segment_cache_key(segment)

            if cache_key in cache:
                segment_score = cache[cache_key]
            else:
                segment_score = self._score_segment(segment)
                cache[cache_key] = segment_score

            total_score += segment_score

        return total_score

    def _get_segment_cache_key(self, segment: Segment) -> str:
        """Generate cache key for segment scoring"""
        if isinstance(segment, (FixedSegment, GroupSegment, OptionalSegment)):
            return f"{type(segment).__name__}:{segment.content}"
        elif isinstance(segment, CharClassSegment):
            return f"{type(segment).__name__}:{segment.value}"
        else:
            return f"{type(segment).__name__}:{str(segment)}"

    def _score_segment(self, segment: Segment) -> float:
        """Calculate score for individual segment"""
        # Prefer fixed segments (higher score)
        if isinstance(segment, FixedSegment):
            return len(segment.content) * 2.0  # Fixed content is valuable

        # Character classes get medium score
        if isinstance(segment, CharClassSegment):
            return 1.0

        # Optional segments get lower score
        if isinstance(segment, OptionalSegment):
            return 0.5

        # Group segments get variable score based on content
        if isinstance(segment, GroupSegment):
            return 1.5

        # Default score
        return 1.0

    def _score_variant(self, variant: List[Segment]) -> float:
        """Calculate score for a variant based on its enumeration potential."""
        score = 0.0

        # Add points for fixed prefix length
        fixed_length = 0
        for segment in variant:
            if isinstance(segment, FixedSegment):
                fixed_length += len(segment.content)
            else:
                break  # Stop at first non-fixed segment

        score += fixed_length * 2  # Fixed prefix is valuable

        # Add points for variable segments with good enumeration potential
        for segment in variant:
            if isinstance(segment, CharClassSegment):
                if segment.value > 0:
                    score += segment.value
                else:
                    # Estimate value based on charset size and length
                    charset_size = len(segment.charset)
                    if charset_size > 0 and charset_size <= 100:
                        score += 1.0 / charset_size  # Smaller charset = higher value

        return score

    def _optimize_variant(self, segments: List[Segment]) -> EnumerationStrategy:
        """Optimize enumeration for single variant."""

        # Find all variable segments (including inside groups)
        def find_vars(segs):
            vars = []
            for seg in segs:
                if isinstance(seg, CharClassSegment):
                    vars.append(seg)
                elif isinstance(seg, (GroupSegment, OptionalSegment)):
                    vars.extend(find_vars(seg.content))
            return vars

        vars = find_vars(segments)

        if not vars:
            return EnumerationStrategy([], segments, 0.0, 1)

        # Calculate enumeration values
        for segment in vars:
            segment.value = self._calculate_value(segment, segments)

        # Select best enumeration combination
        return self._select_combination(vars, segments)

    def _calculate_value(self, segment: CharClassSegment, all_segments: List[Segment]) -> float:
        """Calculate enumeration value for segment."""
        try:
            # Fixed prefix length
            prefix_length = segment.prefix_length

            # Fixed suffix length
            suffix_length = 0
            for i in range(segment.position + 1, len(all_segments)):
                if isinstance(all_segments[i], FixedSegment):
                    suffix_length += len(all_segments[i])

            # Use actual combination count without artificial limits
            combinations = segment.combinations

            # Value calculation with proper mathematical scaling
            prefix_weight = math.log(max(1, prefix_length + 1))
            suffix_weight = math.log(max(1, suffix_length + 1)) * 0.3

            # Use log scaling to handle large combination counts gracefully
            if combinations > 0:
                cost_weight = math.log(combinations)
            else:
                cost_weight = 1.0

            base_value = (prefix_weight + suffix_weight) / max(0.1, cost_weight)

            # Apply priority factor to base value
            priority_factor = self._calculate_priority_factor(segment)
            final_value = base_value * priority_factor

            logger.debug(
                f"Segment value: prefix={prefix_length}, suffix={suffix_length}, "
                f"combinations={combinations}, priority={priority_factor:.2f}, value={final_value:.3f}"
            )

            return final_value

        except Exception as e:
            logger.warning(f"Value calculation failed: {e}")
            return 0.0

    def _select_combination(self, vars: List[CharClassSegment], all_segments: List[Segment]) -> EnumerationStrategy:
        """Select best enumeration combination using current strategy."""
        if not vars:
            return EnumerationStrategy([], all_segments, 0.0, 1)

        # Use strategy to select segments
        selected_segments = self.strategy.select_segments(vars)

        if not selected_segments:
            return EnumerationStrategy([], all_segments, 0.0, 1)

        # Calculate queries and value using strategy
        total_queries, total_value = self.strategy.evaluate_combination(selected_segments)

        logger.info(
            f"Selected strategy ({self.strategy.__class__.__name__}): {len(selected_segments)} segments, "
            f"value={total_value:.3f}, queries={total_queries}"
        )

        return EnumerationStrategy(selected_segments, all_segments, total_value, total_queries)

    def _calculate_priority_factor(self, segment: CharClassSegment) -> float:
        """Calculate priority multiplier based on segment characteristics."""
        factor = 1.0

        # Quantifier-based priority boost
        if segment.has_range():
            factor *= 3.0  # {8,12} - highest priority
        elif segment.is_specific():
            # Fixed length segments are excellent for enumeration
            if segment.min_length >= 16:
                factor *= 4.0  # {16} - very high priority for long fixed segments
            elif segment.min_length >= 8:
                factor *= 3.5  # {8} - high priority for medium fixed segments
            else:
                factor *= 2.5  # {1-7} - good priority for short fixed segments
        elif segment.has_min():
            # Open-ended segments are less ideal for enumeration
            if segment.min_length >= 8:
                factor *= 2.0  # {8,} - decent priority
            else:
                factor *= 1.5  # {1,} or + - lower priority

        # Additional boost for longer minimum lengths
        if segment.min_length >= 16:
            factor *= 1.5  # Extra boost for very long segments
        elif segment.min_length >= 8:
            factor *= 1.2  # Small boost for long segments

        # Character class type adjustment - this is crucial for correct selection
        if segment.is_positive_class():
            # Positive classes like [a-zA-Z0-9]+ are much better for enumeration
            factor *= 3.0  # Strong boost for positive classes
        else:
            # Negative classes like [^\s\/]+ are poor for enumeration
            factor *= 0.2  # Strong penalty for negative classes

        # Character set size adjustment - prefer optimal enumeration sizes
        charset_size = len(segment.charset)
        if 50 <= charset_size <= 70:  # [a-zA-Z0-9] = 62 chars - optimal range
            factor *= 2.0  # Strong boost for optimal enumeration size
        elif 30 <= charset_size < 50:
            factor *= 1.5  # Good size
        elif 10 <= charset_size < 30:
            factor *= 1.2  # Decent size
        elif charset_size < 10:  # \d = 10 chars
            factor *= 0.8  # Small charset, limited enumeration potential
        else:  # Very large charset (like negated classes)
            factor *= 0.3  # Strong penalty for very large charsets

        return factor

    def _calculate_segment_optimal_depth(self, segment: CharClassSegment) -> int:
        """Calculate optimal enumeration depth using current strategy."""
        return self.strategy.calculate_depth(segment)

    def _is_strategy_feasible(self, query_count: int) -> bool:
        """Check if a strategy with given query count is feasible."""
        # Instead of hard limits, use exponential cost function
        if query_count <= 0:
            return False

        # Allow larger query counts but with exponentially decreasing preference
        # This ensures we don't artificially limit but prefer smaller counts
        return query_count <= self.max_queries or (
            query_count <= self.max_queries * 10 and query_count <= 50000  # Reasonable upper bound for practical use
        )

    def _evaluate_combination(self, combo: tuple) -> Tuple[int, float]:
        """Evaluate a combination of segments using current strategy."""
        return self.strategy.evaluate_combination(list(combo))
