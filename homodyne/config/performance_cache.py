"""
Performance Caching System for Homodyne v2 Configuration
=======================================================

Intelligent caching system for configuration validation results, mode resolution,
and expensive validation checks. Provides persistent caching, cache invalidation
strategies, and memory-efficient operations for enterprise workloads.

Key Features:
- Multi-level caching (memory + disk persistent)
- Intelligent cache invalidation based on content hashing
- Memory-aware cache size management
- Cache warming and preloading strategies
- Performance analytics and cache hit rate tracking
- Thread-safe operations for concurrent access
"""

import hashlib
import json
import pickle
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """
    Individual cache entry with metadata.

    Attributes:
        value: Cached value
        timestamp: Creation timestamp
        access_count: Number of times accessed
        last_access: Last access timestamp
        size_bytes: Estimated memory size in bytes
        content_hash: Hash of the input that generated this entry
        validation_level: Level of validation this entry represents
        dependencies: List of dependent cache keys
    """

    value: Any
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    content_hash: str = ""
    validation_level: str = "basic"
    dependencies: List[str] = field(default_factory=list)


@dataclass
class CacheStats:
    """
    Cache performance statistics.

    Attributes:
        hits: Total cache hits
        misses: Total cache misses
        evictions: Number of entries evicted
        memory_usage_bytes: Current memory usage
        total_entries: Current number of cached entries
        hit_rate: Cache hit rate (0.0 to 1.0)
        average_access_time_ms: Average access time in milliseconds
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    total_entries: int = 0
    hit_rate: float = 0.0
    average_access_time_ms: float = 0.0


class PerformanceCache:
    """
    High-performance caching system for configuration operations.

    Provides intelligent caching with multiple strategies:
    - LRU eviction for memory management
    - Content-based cache invalidation
    - Persistent disk caching
    - Performance monitoring and analytics
    """

    def __init__(
        self,
        max_memory_mb: int = 256,
        max_entries: int = 10000,
        persistent_cache_dir: Optional[Path] = None,
        enable_disk_cache: bool = True,
        cache_version: str = "v2.0",
    ):
        """
        Initialize performance cache.

        Args:
            max_memory_mb: Maximum memory usage in MB
            max_entries: Maximum number of entries
            persistent_cache_dir: Directory for persistent cache
            enable_disk_cache: Enable persistent disk caching
            cache_version: Cache version for invalidation
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self.cache_version = cache_version

        # Thread-safe cache storage
        self._lock = RLock()
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()

        # Persistent cache setup
        self.enable_disk_cache = enable_disk_cache
        if persistent_cache_dir:
            self.persistent_cache_dir = Path(persistent_cache_dir)
        else:
            self.persistent_cache_dir = Path.home() / ".homodyne" / "cache"

        if self.enable_disk_cache:
            self.persistent_cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_stats()

        # Performance tracking
        self._access_times: List[float] = []
        self._max_access_times = 1000  # Keep last 1000 access times

        logger.debug(
            f"Performance cache initialized: {max_memory_mb}MB memory, "
            f"{max_entries} entries, disk_cache={'enabled' if enable_disk_cache else 'disabled'}"
        )

    def get(
        self, key: str, validator_func: Optional[Callable] = None
    ) -> Tuple[Any, bool]:
        """
        Get value from cache with performance tracking.

        Args:
            key: Cache key
            validator_func: Optional function to validate cached value

        Returns:
            Tuple of (value, cache_hit_flag)
        """
        start_time = time.perf_counter()

        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]

                # Validate if validator provided
                if validator_func and not validator_func(entry.value):
                    logger.debug(f"Cache validation failed for key: {key}")
                    self._memory_cache.pop(key)
                    self._stats.misses += 1
                    return None, False

                # Update access metadata
                entry.access_count += 1
                entry.last_access = time.time()

                # Move to end (LRU)
                self._memory_cache.move_to_end(key)

                self._stats.hits += 1
                self._update_access_time(start_time)

                logger.debug(f"Memory cache hit: {key}")
                return entry.value, True

            # Check persistent cache if enabled
            if self.enable_disk_cache:
                persistent_value = self._get_from_persistent_cache(key)
                if persistent_value is not None:
                    # Validate if validator provided
                    if validator_func and not validator_func(persistent_value):
                        logger.debug(
                            f"Persistent cache validation failed for key: {key}"
                        )
                        self._remove_from_persistent_cache(key)
                        self._stats.misses += 1
                        return None, False

                    # Add to memory cache for future access
                    self._add_to_memory_cache(key, persistent_value)

                    self._stats.hits += 1
                    self._update_access_time(start_time)

                    logger.debug(f"Persistent cache hit: {key}")
                    return persistent_value, True

            # Cache miss
            self._stats.misses += 1
            self._update_access_time(start_time)

            return None, False

    def put(
        self,
        key: str,
        value: Any,
        content_hash: Optional[str] = None,
        validation_level: str = "basic",
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """
        Store value in cache with metadata.

        Args:
            key: Cache key
            value: Value to cache
            content_hash: Hash of input content for invalidation
            validation_level: Level of validation performed
            dependencies: Dependent cache keys
        """
        if dependencies is None:
            dependencies = []

        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                value=value,
                content_hash=content_hash or "",
                validation_level=validation_level,
                dependencies=dependencies,
                size_bytes=self._estimate_size(value),
            )

            # Add to memory cache
            self._add_to_memory_cache(key, value, entry)

            # Add to persistent cache if enabled
            if self.enable_disk_cache:
                self._save_to_persistent_cache(key, value, entry)

            logger.debug(
                f"Cached value for key: {key} (size: {entry.size_bytes} bytes)"
            )

    def invalidate(self, key: str) -> bool:
        """
        Invalidate specific cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        with self._lock:
            # Remove from memory cache
            memory_removed = key in self._memory_cache
            if memory_removed:
                self._memory_cache.pop(key)

            # Remove from persistent cache
            persistent_removed = False
            if self.enable_disk_cache:
                persistent_removed = self._remove_from_persistent_cache(key)

            if memory_removed or persistent_removed:
                logger.debug(f"Invalidated cache key: {key}")
                return True

            return False

    def invalidate_by_content_hash(self, content_hash: str) -> int:
        """
        Invalidate entries by content hash.

        Args:
            content_hash: Content hash to match

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = []

            # Find entries with matching content hash
            for key, entry in self._memory_cache.items():
                if entry.content_hash == content_hash:
                    keys_to_remove.append(key)

            # Remove found entries
            for key in keys_to_remove:
                self.invalidate(key)

            logger.debug(f"Invalidated {len(keys_to_remove)} entries by content hash")
            return len(keys_to_remove)

    def invalidate_dependencies(self, dependency_key: str) -> int:
        """
        Invalidate all entries that depend on a given key.

        Args:
            dependency_key: Key that other entries depend on

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = []

            # Find entries with this dependency
            for key, entry in self._memory_cache.items():
                if dependency_key in entry.dependencies:
                    keys_to_remove.append(key)

            # Remove found entries
            for key in keys_to_remove:
                self.invalidate(key)

            logger.debug(f"Invalidated {len(keys_to_remove)} dependent entries")
            return len(keys_to_remove)

    def clear_all(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            entry_count = len(self._memory_cache)
            self._memory_cache.clear()

            if self.enable_disk_cache:
                self._clear_persistent_cache()

            self._stats = CacheStats()
            logger.info(f"Cleared all cache entries ({entry_count} entries)")

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            # Update current stats
            self._stats.total_entries = len(self._memory_cache)
            self._stats.memory_usage_bytes = sum(
                entry.size_bytes for entry in self._memory_cache.values()
            )

            total_requests = self._stats.hits + self._stats.misses
            if total_requests > 0:
                self._stats.hit_rate = self._stats.hits / total_requests

            if self._access_times:
                self._stats.average_access_time_ms = (
                    sum(self._access_times) / len(self._access_times) * 1000
                )

            return self._stats

    def warm_cache(self, warm_configs: List[Dict[str, Any]]) -> int:
        """
        Warm cache with common configurations.

        Args:
            warm_configs: List of configurations to pre-validate and cache

        Returns:
            Number of entries warmed
        """
        warmed_count = 0

        logger.info(f"Warming cache with {len(warm_configs)} configurations")

        for config in warm_configs:
            try:
                # Generate cache key
                config_str = json.dumps(config, sort_keys=True)
                content_hash = self._generate_content_hash(config_str)
                cache_key = f"config_validation:{content_hash}"

                # Check if already cached
                _, cache_hit = self.get(cache_key)
                if not cache_hit:
                    # This would typically involve running actual validation
                    # For now, we'll just mark as warmed
                    self.put(
                        cache_key,
                        {"warmed": True, "config": config},
                        content_hash=content_hash,
                        validation_level="warm",
                    )
                    warmed_count += 1

            except Exception as e:
                logger.debug(f"Failed to warm cache entry: {e}")
                continue

        logger.info(f"Cache warmed: {warmed_count} new entries")
        return warmed_count

    def optimize_memory_usage(self) -> Dict[str, int]:
        """
        Optimize memory usage by removing least valuable entries.

        Returns:
            Dictionary with optimization statistics
        """
        with self._lock:
            initial_memory = sum(
                entry.size_bytes for entry in self._memory_cache.values()
            )
            initial_count = len(self._memory_cache)

            # If we're under limits, no optimization needed
            if (
                initial_memory <= self.max_memory_bytes
                and initial_count <= self.max_entries
            ):
                return {
                    "entries_removed": 0,
                    "memory_freed": 0,
                    "initial_entries": initial_count,
                    "final_entries": initial_count,
                }

            # Calculate value score for each entry (access_count / age / size)
            current_time = time.time()
            entry_scores = {}

            for key, entry in self._memory_cache.items():
                age_hours = (current_time - entry.timestamp) / 3600
                size_kb = entry.size_bytes / 1024

                # Higher score = more valuable
                if age_hours > 0 and size_kb > 0:
                    score = entry.access_count / (age_hours * size_kb)
                else:
                    score = entry.access_count

                entry_scores[key] = score

            # Sort by score (lowest first, these will be removed)
            sorted_entries = sorted(entry_scores.items(), key=lambda x: x[1])

            # Remove least valuable entries until under limits
            removed_count = 0
            memory_freed = 0

            for key, _ in sorted_entries:
                if (
                    len(self._memory_cache) <= self.max_entries
                    and sum(e.size_bytes for e in self._memory_cache.values())
                    <= self.max_memory_bytes
                ):
                    break

                entry = self._memory_cache.pop(key)
                memory_freed += entry.size_bytes
                removed_count += 1
                self._stats.evictions += 1

            final_count = len(self._memory_cache)

            logger.info(
                f"Memory optimization: removed {removed_count} entries, "
                f"freed {memory_freed / 1024:.1f} KB"
            )

            return {
                "entries_removed": removed_count,
                "memory_freed": memory_freed,
                "initial_entries": initial_count,
                "final_entries": final_count,
            }

    def _add_to_memory_cache(
        self, key: str, value: Any, entry: Optional[CacheEntry] = None
    ) -> None:
        """Add entry to memory cache with size management."""
        if entry is None:
            entry = CacheEntry(value=value, size_bytes=self._estimate_size(value))

        # Add to cache
        self._memory_cache[key] = entry

        # Check if we need to optimize memory
        current_memory = sum(e.size_bytes for e in self._memory_cache.values())
        if (
            current_memory > self.max_memory_bytes
            or len(self._memory_cache) > self.max_entries
        ):
            self.optimize_memory_usage()

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            if hasattr(obj, "__sizeof__"):
                return obj.__sizeof__()
            elif isinstance(obj, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in obj.items()
                )
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, str):
                return len(obj.encode("utf-8"))
            elif HAS_NUMPY and isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                # Fallback estimate
                return len(str(obj)) * 4
        except Exception:
            return 1024  # Conservative default

    def _generate_content_hash(self, content: str) -> str:
        """Generate content hash for cache invalidation."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _get_from_persistent_cache(self, key: str) -> Optional[Any]:
        """Get value from persistent disk cache."""
        try:
            cache_file = self.persistent_cache_dir / f"{key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.debug(f"Failed to read persistent cache for {key}: {e}")

        return None

    def _save_to_persistent_cache(
        self, key: str, value: Any, entry: CacheEntry
    ) -> None:
        """Save value to persistent disk cache."""
        try:
            cache_file = self.persistent_cache_dir / f"{key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)

            # Save metadata
            metadata_file = self.persistent_cache_dir / f"{key}.meta"
            metadata = {
                "timestamp": entry.timestamp,
                "content_hash": entry.content_hash,
                "validation_level": entry.validation_level,
                "cache_version": self.cache_version,
            }
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

        except Exception as e:
            logger.debug(f"Failed to save persistent cache for {key}: {e}")

    def _remove_from_persistent_cache(self, key: str) -> bool:
        """Remove entry from persistent cache."""
        removed = False
        try:
            cache_file = self.persistent_cache_dir / f"{key}.pkl"
            metadata_file = self.persistent_cache_dir / f"{key}.meta"

            if cache_file.exists():
                cache_file.unlink()
                removed = True

            if metadata_file.exists():
                metadata_file.unlink()

        except Exception as e:
            logger.debug(f"Failed to remove persistent cache for {key}: {e}")

        return removed

    def _clear_persistent_cache(self) -> None:
        """Clear all persistent cache files."""
        try:
            for file in self.persistent_cache_dir.glob("*.pkl"):
                file.unlink()
            for file in self.persistent_cache_dir.glob("*.meta"):
                file.unlink()
        except Exception as e:
            logger.debug(f"Failed to clear persistent cache: {e}")

    def _load_persistent_stats(self) -> None:
        """Load persistent cache statistics."""
        try:
            stats_file = self.persistent_cache_dir / "cache_stats.json"
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    stats_data = json.load(f)

                # Only load if version matches
                if stats_data.get("cache_version") == self.cache_version:
                    self._stats.hits = stats_data.get("hits", 0)
                    self._stats.misses = stats_data.get("misses", 0)
                    self._stats.evictions = stats_data.get("evictions", 0)

        except Exception as e:
            logger.debug(f"Failed to load persistent stats: {e}")

    def _save_persistent_stats(self) -> None:
        """Save cache statistics to disk."""
        try:
            stats_file = self.persistent_cache_dir / "cache_stats.json"
            stats_data = {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "evictions": self._stats.evictions,
                "cache_version": self.cache_version,
                "last_updated": time.time(),
            }
            with open(stats_file, "w") as f:
                json.dump(stats_data, f)

        except Exception as e:
            logger.debug(f"Failed to save persistent stats: {e}")

    def _update_access_time(self, start_time: float) -> None:
        """Update access time tracking."""
        access_time = time.perf_counter() - start_time
        self._access_times.append(access_time)

        # Keep only recent access times
        if len(self._access_times) > self._max_access_times:
            self._access_times.pop(0)

    def __del__(self):
        """Cleanup and save persistent stats on destruction."""
        if self.enable_disk_cache:
            self._save_persistent_stats()


class ValidationResultCache:
    """
    Specialized cache for configuration validation results.

    Provides automatic cache key generation, dependency tracking,
    and validation-specific optimizations.
    """

    def __init__(self, performance_cache: Optional[PerformanceCache] = None):
        """
        Initialize validation result cache.

        Args:
            performance_cache: Underlying performance cache instance
        """
        if performance_cache is None:
            performance_cache = PerformanceCache(
                max_memory_mb=128, max_entries=5000, enable_disk_cache=True
            )

        self.cache = performance_cache
        self.cache_prefix = "validation_result"

    def get_validation_result(
        self, config: Dict[str, Any], validation_level: str = "basic"
    ) -> Tuple[Optional[Any], bool]:
        """
        Get cached validation result for configuration.

        Args:
            config: Configuration dictionary
            validation_level: Level of validation performed

        Returns:
            Tuple of (validation_result, cache_hit_flag)
        """
        cache_key = self._generate_validation_key(config, validation_level)

        # Validator to ensure cached result is still valid
        def validate_cached_result(cached_result: Any) -> bool:
            if not isinstance(cached_result, dict):
                return False

            # Check if result has required fields
            required_fields = ["is_valid", "timestamp"]
            if not all(field in cached_result for field in required_fields):
                return False

            # Check if result is too old (1 hour expiry for validation)
            result_age = time.time() - cached_result.get("timestamp", 0)
            if result_age > 3600:  # 1 hour
                return False

            return True

        return self.cache.get(cache_key, validator_func=validate_cached_result)

    def cache_validation_result(
        self,
        config: Dict[str, Any],
        validation_result: Dict[str, Any],
        validation_level: str = "basic",
    ) -> None:
        """
        Cache validation result for configuration.

        Args:
            config: Configuration dictionary
            validation_result: Validation result to cache
            validation_level: Level of validation performed
        """
        cache_key = self._generate_validation_key(config, validation_level)

        # Add timestamp to result
        result_with_metadata = validation_result.copy()
        result_with_metadata["timestamp"] = time.time()
        result_with_metadata["validation_level"] = validation_level

        # Generate content hash for invalidation
        config_str = json.dumps(config, sort_keys=True)
        content_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        self.cache.put(
            cache_key,
            result_with_metadata,
            content_hash=content_hash,
            validation_level=validation_level,
        )

    def invalidate_config_validation(self, config: Dict[str, Any]) -> int:
        """
        Invalidate all validation results for a configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Number of entries invalidated
        """
        config_str = json.dumps(config, sort_keys=True)
        content_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        return self.cache.invalidate_by_content_hash(content_hash)

    def _generate_validation_key(
        self, config: Dict[str, Any], validation_level: str
    ) -> str:
        """Generate cache key for validation result."""
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        return f"{self.cache_prefix}:{validation_level}:{config_hash}"


# Global cache instances
_global_performance_cache: Optional[PerformanceCache] = None
_global_validation_cache: Optional[ValidationResultCache] = None


def get_performance_cache() -> PerformanceCache:
    """Get global performance cache instance."""
    global _global_performance_cache

    if _global_performance_cache is None:
        # Determine cache settings based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        if available_memory_gb > 16:
            cache_memory_mb = 512
            max_entries = 20000
        elif available_memory_gb > 8:
            cache_memory_mb = 256
            max_entries = 10000
        else:
            cache_memory_mb = 128
            max_entries = 5000

        _global_performance_cache = PerformanceCache(
            max_memory_mb=cache_memory_mb,
            max_entries=max_entries,
            enable_disk_cache=True,
        )

        logger.info(
            f"Global performance cache initialized: {cache_memory_mb}MB, {max_entries} entries"
        )

    return _global_performance_cache


def get_validation_cache() -> ValidationResultCache:
    """Get global validation result cache instance."""
    global _global_validation_cache

    if _global_validation_cache is None:
        _global_validation_cache = ValidationResultCache(get_performance_cache())

    return _global_validation_cache


def clear_all_caches() -> None:
    """Clear all global cache instances."""
    global _global_performance_cache, _global_validation_cache

    if _global_performance_cache:
        _global_performance_cache.clear_all()

    if _global_validation_cache:
        _global_validation_cache.cache.clear_all()

    logger.info("All global caches cleared")
