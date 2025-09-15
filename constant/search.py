#!/usr/bin/env python3

"""
Search Engine Configuration Constants

This module contains constants related to search functionality,
language processing, and search result limits.
"""

from typing import List, Set

# Search result limits
API_MAX_PAGES: int = 20
WEB_MAX_PAGES: int = 25
API_RESULTS_PER_PAGE: int = 100
WEB_RESULTS_PER_PAGE: int = 20
API_LIMIT: int = API_MAX_PAGES * API_RESULTS_PER_PAGE
WEB_LIMIT: int = WEB_MAX_PAGES * WEB_RESULTS_PER_PAGE

# Popular programming languages for search prioritization
POPULAR_LANGUAGES: List[str] = [
    "Python",
    "JavaScript",
    "TypeScript",
    "Java",
    "C#",
    "C++",
    "Go",
    "Rust",
    "PHP",
    "Ruby",
    "Swift",
    "Kotlin",
    "Scala",
    "C",
    "R",
    "MATLAB",
    "Perl",
    "Lua",
    "Haskell",
    "Erlang",
    "Clojure",
    "Assembly",
    "COBOL",
    "Fortran",
    "Pascal",
    "Ada",
    "Prolog",
]

# File size ranges for search filtering
SIZE_RANGES: List[str] = [
    "size:1..1000",  # Small files (1B - 1KB)
    "size:1000..10000",  # Medium files (1KB - 10KB)
    "size:10000..100000",  # Large files (10KB - 100KB)
    "size:>100000",  # Very large files (>100KB)
]

# Allowed search operators for query construction
ALLOWED_OPERATORS: Set[str] = {
    "AND",
    "OR",
    "NOT",
    "in:file",
    "in:path",
    "language:",
    "extension:",
    "size:",
    "user:",
    "repo:",
    "org:",
}
