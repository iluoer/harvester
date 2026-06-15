#!/usr/bin/env python3

"""
Default Configuration Values

This module provides default configuration values for the entire application.
It ensures consistent defaults across all configuration sections.

Key Features:
- Centralized default values
- Complete configuration template
- Easy customization
- Type-safe defaults
- Auto-sync with Config schema
"""

from typing import Any, Dict

from config.schemas import Config


def get_default_config() -> Dict[str, Any]:
    """Get complete default configuration

    This function creates a Config instance with default values and converts it to a dictionary.
    Then it adds example rate_limits and tasks for demonstration purposes.
    This approach ensures automatic synchronization with the Config schema.

    Returns:
        Dict[str, Any]: Default configuration dictionary
    """

    # Convert to dictionary to get the base structure
    config = Config().to_dict()

    # Add example rate limits for demonstration
    config["ratelimits"].update(
        {
            "github_api": {"base_rate": 1.0, "burst_limit": 5, "adaptive": True},
            "github_web": {"base_rate": 2.0, "burst_limit": 3, "adaptive": False},
        }
    )

    # Add example tasks for demonstration
    config["tasks"].extend(
        [
            {
                "name": "openai",
                "enabled": True,
                "provider_type": "openai_like",
                "use_api": False,
                "stages": {
                    "search": True,
                    "gather": True,
                    "check": True,
                    "inspect": True,
                },
                "extras": {},
                "api": {
                    "base_url": "https://api.openai.com",
                    "completion_path": "/v1/chat/completions",
                    "model_path": "/v1/models",
                    "default_model": "gpt-4o-mini",
                    "auth_key": "Authorization",
                    "extra_headers": {},
                    "api_version": "",
                    "timeout": 30,
                    "retries": 3,
                },
                "patterns": {
                    "key_pattern": "sk(?:-proj)?-[a-zA-Z0-9]{20}T3BlbkFJ[a-zA-Z0-9]{20}",
                    "address_pattern": "",
                    "endpoint_pattern": "",
                    "model_pattern": "",
                },
                "conditions": [{"query": '"T3BlbkFJ"'}],
                "rate_limit": {"base_rate": 2.0, "burst_limit": 10, "adaptive": True},
                "storage": {
                    "directory": "",
                    "plan": "",
                },
            }
        ]
    )

    return config
