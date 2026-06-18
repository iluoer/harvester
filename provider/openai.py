#!/usr/bin/env python3

"""
OpenAI provider implementation.
"""

from typing import List

from core.models import Condition

from .openai_like import OpenAILikeProvider
from .registry import register_provider


class OpenAIProvider(OpenAILikeProvider):
    """OpenAI provider implementation."""

    def __init__(self, conditions: List[Condition], **kwargs):
        # Set OpenAI specific defaults
        self.defaults(kwargs, {"name": "openai", "base_url": "https://api.openai.com", "default_model": "gpt-4o-mini"})

        super().__init__(conditions=conditions, **kwargs)


register_provider("openai", OpenAIProvider)
