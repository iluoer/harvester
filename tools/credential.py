#!/usr/bin/env python3

"""
GitHub Credentials Manager

This module manages GitHub session tokens and API tokens with load balancing.
It provides thread-safe access to multiple credentials for improved concurrency.

Key Features:
- Load balanced credential distribution
- Support for both session tokens and API tokens
- Thread-safe credential access
- Usage statistics and monitoring
- Automatic fallback between credential types
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .balancer import Balancer, Strategy


@dataclass
class CredentialStats:
    """Credential usage statistics"""

    total_requests: int
    session_requests: int
    token_requests: int
    sessions_count: int
    tokens_count: int
    session_percentage: float
    token_percentage: float
    session_stats: Optional[Dict] = None
    token_stats: Optional[Dict] = None

    @property
    def has_sessions(self) -> bool:
        """Check if sessions are available"""
        return self.sessions_count > 0

    @property
    def has_tokens(self) -> bool:
        """Check if tokens are available"""
        return self.tokens_count > 0

    @property
    def total_credentials(self) -> int:
        """Total number of credentials"""
        return self.sessions_count + self.tokens_count


class Credentials:
    """GitHub credentials manager with load balancing"""

    def __init__(self, sessions: List[str], tokens: List[str], strategy: str = "round_robin"):
        """Initialize credentials manager

        Args:
            sessions: List of GitHub session tokens
            tokens: List of GitHub API tokens
            strategy: Load balancing strategy ("round_robin" or "random")
        """
        if not sessions and not tokens:
            raise ValueError("At least one session or token must be provided")

        self.sessions = sessions.copy() if sessions else []
        self.tokens = tokens.copy() if tokens else []

        # Convert string strategy to enum
        strategy_enum = Strategy.ROUND_ROBIN if strategy == "round_robin" else Strategy.RANDOM

        # Create balancers for each credential type
        self.session_balancer = Balancer(self.sessions, strategy_enum) if self.sessions else None
        self.token_balancer = Balancer(self.tokens, strategy_enum) if self.tokens else None

        self.lock = threading.Lock()
        self.total_requests = 0
        self.session_requests = 0
        self.token_requests = 0

    def get_session(self) -> Optional[str]:
        """Get next session token

        Returns:
            Optional[str]: Session token or None if no sessions available
        """
        if not self.session_balancer:
            return None

        with self.lock:
            self.total_requests += 1
            self.session_requests += 1
            return self.session_balancer.get()

    def get_token(self) -> Optional[str]:
        """Get next API token

        Returns:
            Optional[str]: API token or None if no tokens available
        """
        if not self.token_balancer:
            return None

        with self.lock:
            self.total_requests += 1
            self.token_requests += 1
            return self.token_balancer.get()

    def get_credential(self, prefer_token: bool = True) -> Tuple[str, str]:
        """Get next credential with type preference

        Args:
            prefer_token: Whether to prefer API tokens over sessions

        Returns:
            Tuple[str, str]: (credential_value, credential_type)

        Raises:
            RuntimeError: If no credentials are available
        """
        if prefer_token and self.token_balancer:
            token = self.get_token()
            if token:
                return token, "token"

        if self.session_balancer:
            session = self.get_session()
            if session:
                return session, "session"

        if not prefer_token and self.token_balancer:
            token = self.get_token()
            if token:
                return token, "token"

        raise RuntimeError("No credentials available")

    def get_any(self) -> Tuple[str, str]:
        """Get any available credential

        Returns:
            Tuple[str, str]: (credential_value, credential_type)

        Raises:
            RuntimeError: If no credentials are available
        """
        return self.get_credential(prefer_token=True)

    def has_sessions(self) -> bool:
        """Check if sessions are available

        Returns:
            bool: True if sessions are available
        """
        return bool(self.sessions)

    def has_tokens(self) -> bool:
        """Check if tokens are available

        Returns:
            bool: True if tokens are available
        """
        return bool(self.tokens)

    def has_credentials(self) -> bool:
        """Check if any credentials are available

        Returns:
            bool: True if any credentials are available
        """
        return self.has_sessions() or self.has_tokens()

    def update_sessions(self, sessions: List[str]) -> None:
        """Update session tokens list

        Args:
            sessions: New list of session tokens
        """
        with self.lock:
            self.sessions = sessions.copy() if sessions else []
            if self.sessions:
                if self.session_balancer:
                    self.session_balancer.update_items(self.sessions)
                else:
                    strategy = self.token_balancer.strategy if self.token_balancer else Strategy.ROUND_ROBIN
                    self.session_balancer = Balancer(self.sessions, strategy)
            else:
                self.session_balancer = None

    def update_tokens(self, tokens: List[str]) -> None:
        """Update API tokens list

        Args:
            tokens: New list of API tokens
        """
        with self.lock:
            self.tokens = tokens.copy() if tokens else []
            if self.tokens:
                if self.token_balancer:
                    self.token_balancer.update_items(self.tokens)
                else:
                    strategy = self.session_balancer.strategy if self.session_balancer else Strategy.ROUND_ROBIN
                    self.token_balancer = Balancer(self.tokens, strategy)
            else:
                self.token_balancer = None

    def reset_stats(self) -> None:
        """Reset usage statistics"""
        with self.lock:
            self.total_requests = 0
            self.session_requests = 0
            self.token_requests = 0
            if self.session_balancer:
                self.session_balancer.reset()
            if self.token_balancer:
                self.token_balancer.reset()

    def get_stats(self) -> CredentialStats:
        """Get usage statistics

        Returns:
            CredentialStats: Comprehensive usage statistics
        """
        with self.lock:
            # Calculate percentages
            if self.total_requests > 0:
                session_percentage = round((self.session_requests / self.total_requests) * 100, 2)
                token_percentage = round((self.token_requests / self.total_requests) * 100, 2)
            else:
                session_percentage = 0.0
                token_percentage = 0.0

            return CredentialStats(
                total_requests=self.total_requests,
                session_requests=self.session_requests,
                token_requests=self.token_requests,
                sessions_count=len(self.sessions),
                tokens_count=len(self.tokens),
                session_percentage=session_percentage,
                token_percentage=token_percentage,
                session_stats=self.session_balancer.get_stats() if self.session_balancer else None,
                token_stats=self.token_balancer.get_stats() if self.token_balancer else None,
            )

    def __str__(self) -> str:
        """String representation

        Returns:
            str: String representation
        """
        return f"Credentials(sessions={len(self.sessions)}, tokens={len(self.tokens)}, requests={self.total_requests})"

    def __repr__(self) -> str:
        """Detailed string representation

        Returns:
            str: Detailed representation
        """
        return f"Credentials(sessions={self.sessions}, tokens={self.tokens})"
