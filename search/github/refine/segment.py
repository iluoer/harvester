#!/usr/bin/env python3

"""
Segment types for regex pattern representation.
"""

from dataclasses import dataclass
from typing import List, Set, Union


@dataclass
class Segment:
    """Base class for regex pattern segments."""

    position: int = 0
    prefix_length: int = 0


@dataclass
class FixedSegment(Segment):
    """Fixed string segment."""

    content: str = ""

    def __len__(self) -> int:
        return len(self.content)


@dataclass
class CharClassSegment(Segment):
    """Character class segment with quantifiers."""

    charset: Set[str] = None
    min_length: int = 1
    max_length: Union[int, float] = 1
    original_quantifier: str = ""  # Store original quantifier (+, *, {n,m})
    original_charset_str: str = ""  # Store original charset string with escapes
    case_sensitive: bool = False  # Whether case sensitive ((?-i) flag)
    value: float = 0.0
    effective_charset: Set[str] = None  # Precomputed effective charset

    def __post_init__(self):
        if self.charset is None:
            self.charset = set()
        if self.effective_charset is None:
            self.effective_charset = set()

    @property
    def combinations(self) -> int:
        """Calculate total possible combinations."""
        if self.max_length == float("inf"):
            # For + and *, set reasonable upper limit
            effective_max = min(50, self.min_length + 20)
        else:
            effective_max = int(self.max_length)

        total = 0
        charset_size = len(self.charset)

        # Avoid infinite loops and overflow
        max_calc_length = min(effective_max, 10)

        for length in range(self.min_length, max_calc_length + 1):
            if charset_size**length > 1000000:  # Prevent overflow
                total += 1000000
                break
            total += charset_size**length

        return min(total, 1000000)  # Cap at reasonable limit

    def has_range(self) -> bool:
        """Check if quantifier has range like {8,12}."""
        if not self.original_quantifier:
            return False
        if self.original_quantifier.startswith("{") and self.original_quantifier.endswith("}"):
            content = self.original_quantifier[1:-1]
            return "," in content and content.split(",")[1].strip() != ""
        return False

    def has_min(self) -> bool:
        """Check if quantifier has minimum like {8,} or +."""
        if not self.original_quantifier:
            return False
        if self.original_quantifier == "+":
            return True
        if self.original_quantifier.startswith("{") and self.original_quantifier.endswith("}"):
            content = self.original_quantifier[1:-1]
            return "," in content and content.split(",")[1].strip() == ""
        return False

    def is_specific(self) -> bool:
        """Check if quantifier is specific like {8}."""
        if not self.original_quantifier:
            return self.min_length == self.max_length == 1
        if self.original_quantifier.startswith("{") and self.original_quantifier.endswith("}"):
            content = self.original_quantifier[1:-1]
            return "," not in content
        return False

    def is_positive_class(self) -> bool:
        """Check if this is a positive character class (not negated)."""
        if not self.original_charset_str:
            return True
        return not (self.original_charset_str.startswith("[^") and self.original_charset_str.endswith("]"))


@dataclass
class OptionalSegment(Segment):
    """Optional segment (?:...)?"""

    content: List[Segment] = None

    def __post_init__(self):
        if self.content is None:
            self.content = []

    def variants(self) -> List[List[Segment]]:
        """Return empty and content variants."""
        return [[], self.content]


@dataclass
class GroupSegment(Segment):
    """Group segment (?:...) or (...)"""

    content: List[Segment] = None
    capturing: bool = False

    def __post_init__(self):
        if self.content is None:
            self.content = []

    def flatten(self) -> List[Segment]:
        """Flatten group content."""
        return self.content


@dataclass
class EnumerationStrategy:
    """Strategy for regex enumeration."""

    segments: List[CharClassSegment]
    original: List[Segment]
    value: float
    queries: int
