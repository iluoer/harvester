#!/usr/bin/env python3

"""
Persistence strategies for different storage modes.

This module defines the strategy pattern implementation for handling
different persistence approaches: simple file-based and shard-based storage.
"""

import json
import os
import shutil
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from state.models import PersistenceMetrics
from tools.logger import get_logger
from core.models import Service

from .atomic import AtomicFileWriter
from .shard import NDJSONShardWriter
from .snapshot import SnapshotManager as BaseSnapshotManager

logger = get_logger("storage")


class PersistenceStrategy(ABC):
    """Abstract base class for persistence strategies.

    Defines the interface that all persistence strategies must implement.
    This allows for clean separation between simple file-based and
    shard-based persistence approaches.
    """

    def __init__(self, directory: str, files: Dict[str, str]):
        """Initialize strategy with directory and file mappings.

        Args:
            directory: Base directory for storage
            files: Mapping of result types to file paths
        """
        self.directory = directory
        self.files = files

    @abstractmethod
    def write_data(self, result_type: str, items: List[Any], stats: PersistenceMetrics) -> None:
        """Write data items to storage.

        Args:
            result_type: Type of result being stored
            items: List of items to store
            stats: Metrics object to update
        """
        pass

    @abstractmethod
    def supports_snapshots(self) -> bool:
        """Check if this strategy supports snapshot generation.

        Returns:
            True if snapshots are supported, False otherwise
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup strategy resources."""
        pass


class SimpleFileStrategy(PersistenceStrategy):
    """Simple text file persistence strategy.

    Stores data as plain text files, one item per line.
    Does not support snapshots or advanced features.
    """

    def write_data(self, result_type: str, items: List[Any], stats: PersistenceMetrics) -> None:
        """Write items to simple text file.

        Args:
            result_type: Type of result being stored
            items: List of items to store
            stats: Metrics object to update
        """
        if not items:
            return

        filepath = self.files.get(result_type)
        if not filepath:
            logger.error(f"No file path configured for result type: {result_type}")
            return

        try:
            # Convert items to text lines with minimal fields
            lines = []
            for item in items:
                if isinstance(item, Service):
                    # Build minimal JSON: only endpoint, key, and selected meta
                    base: Dict[str, Any] = {}
                    if item.endpoint:
                        base["endpoint"] = item.endpoint
                    if item.key:
                        base["key"] = item.key
                    # paid/balance
                    try:
                        if isinstance(item.meta, dict):
                            if "paid" in item.meta:
                                base["paid"] = bool(item.meta["paid"]) if not isinstance(item.meta["paid"], bool) else item.meta["paid"]
                            if "balance" in item.meta:
                                bal = item.meta["balance"]
                                bal_val: float | None = None
                                try:
                                    if isinstance(bal, (int, float)):
                                        bal_val = float(bal)
                                    elif isinstance(bal, str) and bal.strip().lstrip("-+").isdigit():
                                        bal_val = float(int(bal.strip()))
                                except Exception:
                                    bal_val = None
                                if bal_val is not None:
                                    # Convert cents to yuan, 2 decimals
                                    base["balance"] = round(bal_val / 100.0, 2)
                                else:
                                    # If already looks like yuan (float string), try to format
                                    try:
                                        base["balance"] = round(float(bal), 2)
                                    except Exception:
                                        base["balance"] = bal
                    except Exception:
                        pass
                    lines.append(json.dumps(base, ensure_ascii=False))
                elif hasattr(item, "serialize"):
                    # Fallback: parse serialized if possible and strip extras
                    try:
                        data = json.loads(item.serialize())
                        minimal = {}
                        if isinstance(data, dict):
                            for k in ("endpoint", "key", "paid", "balance"):
                                if k in data and data[k] is not None and data[k] != "":
                                    minimal[k] = data[k]
                        # Normalize balance if needed
                        if "balance" in minimal and not isinstance(minimal["balance"], (int, float)):
                            try:
                                sval = str(minimal["balance"]).strip()
                                if sval.lstrip("-+").isdigit():
                                    minimal["balance"] = round(int(sval) / 100.0, 2)
                                else:
                                    minimal["balance"] = round(float(sval), 2)
                            except Exception:
                                pass
                        lines.append(json.dumps(minimal or data, ensure_ascii=False))
                    except Exception:
                        lines.append(str(item))
                else:
                    lines.append(str(item))

            # No implicit .bak creation to keep provider folders clean (per product decision)

            # Write to file atomically
            AtomicFileWriter.append_atomic(filepath, lines)

            logger.info(f"Saved {len(lines)} {result_type} items to simple file")

        except Exception as e:
            logger.error(f"Failed to write {result_type} to simple file: {e}")

    def supports_snapshots(self) -> bool:
        """Simple files do not support snapshots."""
        return False

    def cleanup(self) -> None:
        """No cleanup needed for simple files."""
        pass


class ShardStrategy(PersistenceStrategy):
    """NDJSON shard persistence strategy.

    Stores data as NDJSON shard files with indexing and rotation.
    Supports snapshot generation and advanced features.
    """

    def __init__(self, directory: str, files: Dict[str, str]):
        """Initialize shard strategy.

        Args:
            directory: Base directory for storage
            files: Mapping of result types to file paths
        """
        super().__init__(directory, files)
        self._shard_writers: Dict[str, NDJSONShardWriter] = {}
        self._lock = threading.Lock()

    def write_data(self, result_type: str, items: List[Any], stats: PersistenceMetrics) -> None:
        """Write items to NDJSON shard files.

        Args:
            result_type: Type of result being stored
            items: List of items to store
            stats: Metrics object to update
        """
        if not items:
            return

        try:
            # Convert items to NDJSON records (minimal fields for key files)
            records: List[Dict[str, Any]] = []
            minimal_types = {"valid", "invalid", "no_quota", "wait_check"}
            for item in items:
                if isinstance(item, Service):
                    if result_type in minimal_types:
                        rec: Dict[str, Any] = {"key": item.key}
                        if item.endpoint:
                            rec["endpoint"] = item.endpoint
                        if item.address:
                            rec["address"] = item.address
                        if item.model:
                            rec["model"] = item.model
                        records.append(rec)
                    else:
                        # Keep full serialize for non-key result types
                        try:
                            records.append(json.loads(item.serialize()))
                        except Exception:
                            records.append({"value": item.serialize()})
                elif hasattr(item, "serialize"):
                    try:
                        data = json.loads(item.serialize())
                        if result_type in minimal_types and isinstance(data, dict):
                            rec: Dict[str, Any] = {}
                            for k in ("key", "endpoint", "address", "model"):
                                if k in data and data[k]:
                                    rec[k] = data[k]
                            records.append(rec or data)
                        else:
                            records.append(data)
                    except Exception:
                        records.append({"value": str(item)})
                else:
                    records.append({"value": str(item)})

            # Write to shard
            writer = self._get_shard_writer(result_type)
            writer.append_records(records, stats)

            logger.info(f"Saved {len(records)} {result_type} items to shard")

        except Exception as e:
            logger.error(f"Failed to write {result_type} to shard: {e}")

    def supports_snapshots(self) -> bool:
        """Shard files support snapshots."""
        return True

    def cleanup(self) -> None:
        """Cleanup shard writers and resources."""
        with self._lock:
            self._shard_writers.clear()

    def _get_shard_writer(self, result_type: str) -> NDJSONShardWriter:
        """Get or create shard writer for result type.

        Args:
            result_type: Type of result

        Returns:
            NDJSONShardWriter instance
        """
        with self._lock:
            writer = self._shard_writers.get(result_type)
            if not writer:
                shard_root = os.path.join(self.directory, "shards")
                writer = NDJSONShardWriter(shard_root, result_type)
                self._shard_writers[result_type] = writer
            return writer


class SnapshotManager:
    """Dedicated snapshot management with periodic building support.

    Manages the lifecycle of snapshot generation including periodic
    background building and proper cleanup.
    """

    def __init__(self, directory: str, result_types: List[str], provider_name: str):
        """Initialize snapshot manager.

        Args:
            directory: Base directory containing shards
            result_types: List of result types to manage
            provider_name: Name of the provider for logging
        """
        self.directory = directory
        self.result_types = result_types
        self.provider_name = provider_name

        # Thread management
        self._periodic_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Statistics
        self.stats = {"last_snapshot": 0.0, "snapshot_count": 0, "total_snapshot_time": 0.0, "snapshot_operations": 0}

    def build_snapshot(self, result_type: str) -> int:
        """Build snapshot for specific result type.

        Args:
            result_type: Type of result to build snapshot for

        Returns:
            Number of records in the snapshot
        """
        shard_root = os.path.join(self.directory, "shards", result_type)
        if not os.path.isdir(shard_root):
            logger.debug(f"No shard directory for {result_type} in {self.provider_name}")
            return 0

        snapshots_dir = os.path.join(self.directory, "snapshots")
        os.makedirs(snapshots_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshots_dir, f"{result_type}.json")

        try:
            manager = BaseSnapshotManager(shard_root, snapshot_path)
            start_time = time.time()
            count = manager.build_snapshot()
            duration = time.time() - start_time

            # Update statistics
            with self._lock:
                self.stats["last_snapshot"] = time.time()
                self.stats["snapshot_count"] += 1
                self.stats["total_snapshot_time"] += duration
                self.stats["snapshot_operations"] += 1

            logger.info(f"Built snapshot for {result_type} with {count} records in {duration:.3f}s")
            return count

        except Exception as e:
            logger.error(f"Failed to build snapshot for {result_type}: {e}")
            return 0

    def build_all_snapshots(self) -> Dict[str, int]:
        """Build snapshots for all result types.

        Returns:
            Dictionary mapping result types to record counts
        """
        results = {}
        for result_type in self.result_types:
            try:
                results[result_type] = self.build_snapshot(result_type)
            except Exception as e:
                logger.error(f"Failed to build snapshot for {result_type}: {e}")
                results[result_type] = 0
        return results

    def start_periodic(self, interval_sec: int = 300) -> None:
        """Start periodic snapshot building.

        Args:
            interval_sec: Interval between snapshots in seconds
        """
        with self._lock:
            if self._periodic_thread and self._periodic_thread.is_alive():
                logger.debug(f"Periodic snapshot already running for {self.provider_name}")
                return

            self._running = True
            self._periodic_thread = threading.Thread(
                target=self._periodic_loop, args=(interval_sec,), daemon=True, name=f"snapshot-{self.provider_name}"
            )
            self._periodic_thread.start()
            logger.info(f"Started periodic snapshots for {self.provider_name} (interval: {interval_sec}s)")

    def stop(self) -> None:
        """Stop periodic snapshot building and cleanup."""
        with self._lock:
            if not self._running:
                return

            self._running = False

            if self._periodic_thread and self._periodic_thread.is_alive():
                # Wait for thread to finish
                self._periodic_thread.join(timeout=5.0)
                if self._periodic_thread.is_alive():
                    logger.warning(f"Snapshot thread for {self.provider_name} did not stop gracefully")

            self._periodic_thread = None
            logger.info(f"Stopped periodic snapshots for {self.provider_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get snapshot statistics.

        Returns:
            Dictionary with snapshot statistics
        """
        with self._lock:
            return self.stats.copy()

    def _periodic_loop(self, interval_sec: int) -> None:
        """Periodic snapshot building loop.

        Args:
            interval_sec: Interval between snapshots
        """
        while self._running:
            try:
                time.sleep(interval_sec)
                if not self._running:
                    break

                self.build_all_snapshots()

            except Exception as e:
                logger.error(f"Error in periodic snapshot for {self.provider_name}: {e}")
