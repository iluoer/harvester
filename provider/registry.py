#!/usr/bin/env python3

"""
Global provider registry for dynamic provider management.
Enables plugin-style provider registration and creation.
"""

from typing import Dict, List, Type

from core.models import Condition
from core.types import IProvider
from tools.logger import get_logger

logger = get_logger("registry")


class ProviderRegistry:
    """Global registry for provider types and creation"""

    _registry: Dict[str, Type[IProvider]] = {}

    @classmethod
    def register(cls, provider_type: str, provider_class: Type[IProvider]) -> None:
        """Register a provider class for a given type

        Args:
            provider_type: Provider type identifier
            provider_class: Provider class to register
        """
        if not provider_type:
            raise ValueError("Provider type cannot be empty")

        if not issubclass(provider_class, IProvider):
            raise ValueError(f"Provider class must inherit from Provider interface")

        cls._registry[provider_type.lower()] = provider_class
        logger.debug(f"Registered provider: {provider_type} -> {provider_class.__name__}")

    @classmethod
    def create(cls, provider_type: str, conditions: List[Condition], **kwargs) -> IProvider:
        """Create provider instance by type

        Args:
            provider_type: Provider type identifier
            conditions: Search conditions for provider
            **kwargs: Additional provider-specific parameters

        Returns:
            Provider: Configured provider instance

        Raises:
            ValueError: If provider type is not registered
        """
        provider_type = provider_type.lower()

        if provider_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown provider type: {provider_type}. Available: {available}")

        provider_class = cls._registry[provider_type]

        try:
            # All providers now use unified **kwargs approach
            return provider_class(conditions=conditions, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create provider {provider_type}: {e}")
            raise

    @classmethod
    def is_registered(cls, provider_type: str) -> bool:
        """Check if provider type is registered

        Args:
            provider_type: Provider type to check

        Returns:
            bool: True if provider type is registered
        """
        return provider_type.lower() in cls._registry

    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get list of registered provider types

        Returns:
            List[str]: List of registered provider type names
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (mainly for testing)"""
        cls._registry.clear()
        logger.debug("Cleared provider registry")


# Convenience function for external use
def register_provider(provider_type: str, provider_class: Type[IProvider]) -> None:
    """Register a provider class

    Args:
        provider_type: Provider type identifier
        provider_class: Provider class to register
    """
    ProviderRegistry.register(provider_type, provider_class)


def create_provider(provider_type: str, conditions: List[Condition], **kwargs) -> IProvider:
    """Create provider instance

    Args:
        provider_type: Provider type identifier
        conditions: Search conditions for provider
        **kwargs: Additional provider-specific parameters

    Returns:
        Provider: Configured provider instance
    """
    return ProviderRegistry.create(provider_type, conditions, **kwargs)


def get_available_providers() -> List[str]:
    """Get list of available provider types

    Returns:
        List[str]: List of available provider type names
    """
    return ProviderRegistry.get_registered_types()
