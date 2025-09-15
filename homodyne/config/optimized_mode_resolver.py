"""
Optimized Mode Resolution for Large Datasets
===========================================

High-performance mode resolution system designed for enterprise workloads
with streaming processing, memory optimization, and scalable algorithms.

Key Features:
- Streaming analysis for very large phi angle arrays (billions of data points)
- Lazy evaluation with intelligent caching
- Memory-efficient processing with configurable batch sizes
- Parallel processing for multi-configuration scenarios
- Progressive refinement of mode suggestions
- Real-time memory usage monitoring
"""

import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import psutil

try:
    from numba import jit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

from homodyne.config.mode_resolver import ModeCompatibilityResult, ModeResolver
from homodyne.config.performance_cache import get_performance_cache
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StreamingConfig:
    """
    Configuration for streaming processing.

    Attributes:
        chunk_size: Number of elements per processing chunk
        max_memory_mb: Maximum memory usage in MB
        enable_parallel: Enable parallel chunk processing
        num_workers: Number of worker threads
        cache_intermediate: Cache intermediate results
        progress_callback: Optional progress reporting callback
    """

    chunk_size: int = 100000
    max_memory_mb: int = 512
    enable_parallel: bool = True
    num_workers: int = 4
    cache_intermediate: bool = True
    progress_callback: Optional[Callable[[float, str], None]] = None


@dataclass
class AnalysisMetrics:
    """
    Performance metrics for mode analysis.

    Attributes:
        processing_time_ms: Total processing time in milliseconds
        memory_peak_mb: Peak memory usage in MB
        chunks_processed: Number of chunks processed
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        data_points_analyzed: Total data points analyzed
        mode_confidence: Final mode confidence score
    """

    processing_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    chunks_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    data_points_analyzed: int = 0
    mode_confidence: float = 0.0


class OptimizedModeResolver(ModeResolver):
    """
    High-performance mode resolver optimized for large datasets.

    Extends the base ModeResolver with streaming capabilities,
    memory optimization, and performance monitoring.
    """

    def __init__(self, streaming_config: Optional[StreamingConfig] = None):
        """
        Initialize optimized mode resolver.

        Args:
            streaming_config: Configuration for streaming processing
        """
        super().__init__()

        if streaming_config is None:
            # Auto-configure based on system memory
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            if available_memory_gb > 16:
                streaming_config = StreamingConfig(
                    chunk_size=1000000,  # 1M elements
                    max_memory_mb=1024,
                    num_workers=8,
                )
            elif available_memory_gb > 8:
                streaming_config = StreamingConfig(
                    chunk_size=500000,  # 500K elements
                    max_memory_mb=512,
                    num_workers=4,
                )
            else:
                streaming_config = StreamingConfig(
                    chunk_size=100000,  # 100K elements
                    max_memory_mb=256,
                    num_workers=2,
                )

        self.streaming_config = streaming_config
        self.cache = get_performance_cache()
        self._processing_lock = Lock()

        # Enhanced thresholds for large datasets
        self.large_dataset_thresholds = {
            "very_large": 100_000_000,  # 100M points
            "large": 10_000_000,  # 10M points
            "medium": 1_000_000,  # 1M points
            "small": 100_000,  # 100K points
        }

        logger.debug(
            f"Optimized mode resolver initialized: chunk_size={streaming_config.chunk_size}, "
            f"max_memory={streaming_config.max_memory_mb}MB, workers={streaming_config.num_workers}"
        )

    def resolve_mode_streaming(
        self,
        config: Dict[str, Any],
        data_dict: Optional[Dict[str, Any]] = None,
        cli_args: Optional[Any] = None,
    ) -> Tuple[str, AnalysisMetrics]:
        """
        Resolve mode with streaming analysis and performance metrics.

        Args:
            config: Configuration dictionary
            data_dict: Data dictionary (potentially very large)
            cli_args: CLI arguments

        Returns:
            Tuple of (resolved_mode, analysis_metrics)
        """
        start_time = time.perf_counter()
        initial_memory = psutil.Process().memory_info().rss / (1024**2)

        metrics = AnalysisMetrics()

        # Try standard resolution first (fast path)
        resolved_mode = self._try_fast_resolution(config, cli_args)
        if resolved_mode:
            metrics.processing_time_ms = (time.perf_counter() - start_time) * 1000
            metrics.memory_peak_mb = (
                psutil.Process().memory_info().rss / (1024**2) - initial_memory
            )
            return resolved_mode, metrics

        # Fall back to streaming analysis for large datasets
        if data_dict:
            resolved_mode, metrics = self._resolve_with_streaming_analysis(
                config, data_dict, start_time, initial_memory
            )
        else:
            # No data available, use fallback
            resolved_mode = "static_isotropic"
            metrics.processing_time_ms = (time.perf_counter() - start_time) * 1000
            metrics.memory_peak_mb = (
                psutil.Process().memory_info().rss / (1024**2) - initial_memory
            )

        logger.info(
            f"Mode resolved: {resolved_mode} (time: {metrics.processing_time_ms:.1f}ms, "
            f"memory: {metrics.memory_peak_mb:.1f}MB, confidence: {metrics.mode_confidence:.2f})"
        )

        return resolved_mode, metrics

    def analyze_large_phi_array(
        self, phi_angles: np.ndarray, detailed_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze very large phi angle arrays with streaming processing.

        Args:
            phi_angles: Large phi angle array
            detailed_analysis: Perform detailed statistical analysis

        Returns:
            Comprehensive analysis results
        """
        if phi_angles.size == 0:
            return {
                "mode_suggestion": "static_isotropic",
                "confidence": 0.0,
                "analysis": {},
            }

        # Generate cache key
        array_hash = self._generate_array_hash(phi_angles)
        cache_key = f"phi_analysis:{array_hash}:{detailed_analysis}"

        # Check cache first
        cached_result, cache_hit = self.cache.get(cache_key)
        if cache_hit:
            logger.debug("Using cached phi angle analysis")
            return cached_result

        start_time = time.perf_counter()

        # Determine processing strategy based on array size
        if phi_angles.size <= self.large_dataset_thresholds["small"]:
            # Small array - process in memory
            result = self._analyze_small_phi_array(phi_angles, detailed_analysis)
        elif phi_angles.size <= self.large_dataset_thresholds["large"]:
            # Medium/large array - use chunked processing
            result = self._analyze_chunked_phi_array(phi_angles, detailed_analysis)
        else:
            # Very large array - use streaming processing
            result = self._analyze_streaming_phi_array(phi_angles, detailed_analysis)

        processing_time = (time.perf_counter() - start_time) * 1000
        result["processing_time_ms"] = processing_time

        # Cache the result
        self.cache.put(cache_key, result, content_hash=array_hash)

        logger.debug(
            f"Phi angle analysis completed: {phi_angles.size:,} angles, "
            f"{processing_time:.1f}ms, mode: {result.get('mode_suggestion')}"
        )

        return result

    def batch_resolve_modes(
        self,
        config_data_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        max_workers: Optional[int] = None,
    ) -> List[Tuple[str, AnalysisMetrics]]:
        """
        Resolve modes for multiple configurations in parallel.

        Args:
            config_data_pairs: List of (config, data_dict) tuples
            max_workers: Maximum number of worker threads

        Returns:
            List of (resolved_mode, metrics) tuples
        """
        if max_workers is None:
            max_workers = self.streaming_config.num_workers

        logger.info(
            f"Batch resolving modes for {len(config_data_pairs)} configurations "
            f"with {max_workers} workers"
        )

        start_time = time.perf_counter()
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.resolve_mode_streaming, config, data_dict): i
                for i, (config, data_dict) in enumerate(config_data_pairs)
            }

            # Collect results in order
            results = [None] * len(config_data_pairs)
            completed = 0

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                    completed += 1

                    # Progress reporting
                    if self.streaming_config.progress_callback:
                        progress = completed / len(config_data_pairs)
                        self.streaming_config.progress_callback(
                            progress,
                            f"Completed {completed}/{len(config_data_pairs)} configurations",
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to resolve mode for configuration {index}: {e}"
                    )
                    results[index] = ("static_isotropic", AnalysisMetrics())

        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Batch mode resolution completed: {len(config_data_pairs)} configs "
            f"in {total_time:.1f}ms ({total_time / len(config_data_pairs):.1f}ms avg)"
        )

        return results

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage and return optimization statistics.

        Returns:
            Dictionary with optimization statistics
        """
        initial_memory = psutil.Process().memory_info().rss / (1024**2)

        # Force garbage collection
        gc.collect()

        # Optimize cache
        cache_stats = self.cache.optimize_memory_usage()

        final_memory = psutil.Process().memory_info().rss / (1024**2)
        memory_freed = max(0, initial_memory - final_memory)

        stats = {
            "memory_freed_mb": memory_freed,
            "cache_optimization": cache_stats,
            "gc_collected": gc.get_count(),
        }

        logger.info(
            f"Memory optimization: freed {memory_freed:.1f}MB, "
            f"cache freed {cache_stats['memory_freed'] / (1024**2):.1f}MB"
        )

        return stats

    def _try_fast_resolution(
        self, config: Dict[str, Any], cli_args: Optional[Any]
    ) -> Optional[str]:
        """
        Try fast resolution methods before expensive analysis.

        Args:
            config: Configuration dictionary
            cli_args: CLI arguments

        Returns:
            Resolved mode or None if needs further analysis
        """
        # Check CLI args first (highest priority)
        cli_mode = self._resolve_from_cli(cli_args)
        if cli_mode:
            return cli_mode

        # Check config explicit mode
        config_mode = self._resolve_from_config(config)
        if config_mode and config_mode != "auto-detect":
            return config_mode

        return None

    def _resolve_with_streaming_analysis(
        self,
        config: Dict[str, Any],
        data_dict: Dict[str, Any],
        start_time: float,
        initial_memory: float,
    ) -> Tuple[str, AnalysisMetrics]:
        """
        Resolve mode using streaming data analysis.

        Args:
            config: Configuration dictionary
            data_dict: Data dictionary
            start_time: Processing start time
            initial_memory: Initial memory usage

        Returns:
            Tuple of (resolved_mode, analysis_metrics)
        """
        metrics = AnalysisMetrics()

        # Analyze phi angles with streaming
        phi_angles = data_dict.get("phi_angles")
        if phi_angles is not None:
            phi_analysis = self.analyze_large_phi_array(
                phi_angles, detailed_analysis=True
            )

            mode_suggestion = phi_analysis.get("mode_suggestion", "static_isotropic")
            metrics.mode_confidence = phi_analysis.get("confidence", 0.5)
            metrics.data_points_analyzed = (
                len(phi_angles) if hasattr(phi_angles, "__len__") else 0
            )

            # Validate mode against data
            if self._validate_mode_compatibility(mode_suggestion, data_dict, config):
                resolved_mode = mode_suggestion
            else:
                # Fall back to safe default
                resolved_mode = "static_isotropic"
                metrics.mode_confidence *= 0.5
        else:
            # No phi angle data, analyze other data characteristics
            data_analysis = self._analyze_data_structure_streaming(data_dict)
            resolved_mode = data_analysis.get("suggested_mode", "static_isotropic")
            metrics.mode_confidence = data_analysis.get("confidence", 0.3)

        # Update metrics
        metrics.processing_time_ms = (time.perf_counter() - start_time) * 1000
        current_memory = psutil.Process().memory_info().rss / (1024**2)
        metrics.memory_peak_mb = max(0, current_memory - initial_memory)

        return resolved_mode, metrics

    @jit(nopython=True if HAS_NUMBA else False)
    def _fast_angle_statistics(
        self, phi_angles: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Fast computation of angle statistics using JIT compilation.

        Args:
            phi_angles: Array of phi angles

        Returns:
            Tuple of (min_angle, max_angle, mean_angle, std_angle)
        """
        if phi_angles.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        min_angle = np.min(phi_angles)
        max_angle = np.max(phi_angles)
        mean_angle = np.mean(phi_angles)
        std_angle = np.std(phi_angles)

        return min_angle, max_angle, mean_angle, std_angle

    def _analyze_small_phi_array(
        self, phi_angles: np.ndarray, detailed_analysis: bool
    ) -> Dict[str, Any]:
        """
        Analyze small phi angle array in memory.

        Args:
            phi_angles: Phi angle array
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            Analysis results
        """
        n_angles = len(phi_angles)

        if n_angles == 1:
            return {
                "mode_suggestion": "static_isotropic",
                "confidence": 0.95,
                "analysis": {
                    "n_angles": n_angles,
                    "reasoning": ["Single phi angle detected"],
                },
            }

        # Convert to radians if needed
        phi_rad = self._ensure_radians(phi_angles)

        # Fast statistics
        min_angle, max_angle, mean_angle, std_angle = self._fast_angle_statistics(
            phi_rad
        )
        angle_range = max_angle - min_angle

        analysis = {
            "n_angles": n_angles,
            "angle_range_rad": angle_range,
            "angle_range_deg": np.degrees(angle_range),
            "mean_angle": mean_angle,
            "std_angle": std_angle,
        }

        # Mode suggestion logic
        if angle_range >= np.pi:  # >= 180 degrees
            mode_suggestion = "laminar_flow"
            confidence = 0.8
            reasoning = ["Wide angle range suggests flow analysis appropriate"]
        elif angle_range >= np.pi / 4:  # >= 45 degrees
            mode_suggestion = "static_anisotropic"
            confidence = 0.7
            reasoning = ["Moderate angle range suggests anisotropic analysis"]
        else:
            mode_suggestion = "static_isotropic"
            confidence = 0.6
            reasoning = ["Limited angle range suggests isotropic analysis"]

        if detailed_analysis:
            # Additional detailed statistics
            analysis.update(self._compute_detailed_angle_statistics(phi_rad))

        analysis["reasoning"] = reasoning

        return {
            "mode_suggestion": mode_suggestion,
            "confidence": confidence,
            "analysis": analysis,
        }

    def _analyze_chunked_phi_array(
        self, phi_angles: np.ndarray, detailed_analysis: bool
    ) -> Dict[str, Any]:
        """
        Analyze medium/large phi angle array using chunked processing.

        Args:
            phi_angles: Phi angle array
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            Analysis results
        """
        n_angles = len(phi_angles)
        chunk_size = min(self.streaming_config.chunk_size, n_angles // 4)

        if chunk_size <= 0:
            chunk_size = n_angles

        logger.debug(f"Analyzing {n_angles:,} angles in chunks of {chunk_size:,}")

        # Process in chunks to manage memory
        chunks_processed = 0
        accumulated_stats = {
            "min_angle": float("inf"),
            "max_angle": float("-inf"),
            "sum_angles": 0.0,
            "sum_squared": 0.0,
            "count": 0,
        }

        for i in range(0, n_angles, chunk_size):
            end_idx = min(i + chunk_size, n_angles)
            chunk = phi_angles[i:end_idx]

            # Convert chunk to radians
            chunk_rad = self._ensure_radians(chunk)

            # Update accumulated statistics
            min_chunk, max_chunk, mean_chunk, _ = self._fast_angle_statistics(chunk_rad)

            accumulated_stats["min_angle"] = min(
                accumulated_stats["min_angle"], min_chunk
            )
            accumulated_stats["max_angle"] = max(
                accumulated_stats["max_angle"], max_chunk
            )
            accumulated_stats["sum_angles"] += np.sum(chunk_rad)
            accumulated_stats["sum_squared"] += np.sum(chunk_rad**2)
            accumulated_stats["count"] += len(chunk_rad)

            chunks_processed += 1

            # Memory management
            if chunks_processed % 10 == 0:
                gc.collect()

        # Compute final statistics
        mean_angle = accumulated_stats["sum_angles"] / accumulated_stats["count"]
        variance = (
            accumulated_stats["sum_squared"] / accumulated_stats["count"]
        ) - mean_angle**2
        std_angle = np.sqrt(max(0, variance))  # Ensure non-negative

        angle_range = accumulated_stats["max_angle"] - accumulated_stats["min_angle"]

        analysis = {
            "n_angles": n_angles,
            "chunks_processed": chunks_processed,
            "angle_range_rad": angle_range,
            "angle_range_deg": np.degrees(angle_range),
            "mean_angle": mean_angle,
            "std_angle": std_angle,
        }

        # Mode suggestion based on accumulated statistics
        if angle_range >= np.pi:
            mode_suggestion = "laminar_flow"
            confidence = 0.75
            reasoning = ["Large angle range from chunked analysis"]
        elif angle_range >= np.pi / 3:
            mode_suggestion = "static_anisotropic"
            confidence = 0.7
            reasoning = ["Moderate angle range from chunked analysis"]
        else:
            mode_suggestion = "static_isotropic"
            confidence = 0.65
            reasoning = ["Limited angle range from chunked analysis"]

        analysis["reasoning"] = reasoning

        return {
            "mode_suggestion": mode_suggestion,
            "confidence": confidence,
            "analysis": analysis,
        }

    def _analyze_streaming_phi_array(
        self, phi_angles: np.ndarray, detailed_analysis: bool
    ) -> Dict[str, Any]:
        """
        Analyze very large phi angle array using streaming processing.

        Args:
            phi_angles: Very large phi angle array
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            Analysis results
        """
        n_angles = len(phi_angles)

        logger.info(f"Streaming analysis of {n_angles:,} phi angles")

        # Use streaming statistics to avoid loading entire array into memory
        if self.streaming_config.enable_parallel:
            return self._parallel_streaming_analysis(phi_angles, detailed_analysis)
        else:
            return self._sequential_streaming_analysis(phi_angles, detailed_analysis)

    def _parallel_streaming_analysis(
        self, phi_angles: np.ndarray, detailed_analysis: bool
    ) -> Dict[str, Any]:
        """
        Parallel streaming analysis of phi angles.

        Args:
            phi_angles: Phi angle array
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            Analysis results
        """
        n_angles = len(phi_angles)
        chunk_size = self.streaming_config.chunk_size
        num_chunks = (n_angles + chunk_size - 1) // chunk_size

        logger.debug(
            f"Parallel processing {num_chunks} chunks with {self.streaming_config.num_workers} workers"
        )

        # Statistics accumulator with thread safety
        global_stats = {
            "min_angle": float("inf"),
            "max_angle": float("-inf"),
            "sum_angles": 0.0,
            "sum_squared": 0.0,
            "count": 0,
            "chunks_processed": 0,
        }
        stats_lock = Lock()

        def process_chunk(chunk_idx: int) -> Dict[str, float]:
            """Process a single chunk and return statistics."""
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_angles)

            chunk = phi_angles[start_idx:end_idx]
            chunk_rad = self._ensure_radians(chunk)

            min_chunk, max_chunk, mean_chunk, _ = self._fast_angle_statistics(chunk_rad)

            return {
                "min_angle": min_chunk,
                "max_angle": max_chunk,
                "sum_angles": np.sum(chunk_rad),
                "sum_squared": np.sum(chunk_rad**2),
                "count": len(chunk_rad),
            }

        # Process chunks in parallel
        with ThreadPoolExecutor(
            max_workers=self.streaming_config.num_workers
        ) as executor:
            chunk_futures = {
                executor.submit(process_chunk, i): i for i in range(num_chunks)
            }

            for future in as_completed(chunk_futures):
                chunk_stats = future.result()

                # Update global statistics thread-safely
                with stats_lock:
                    global_stats["min_angle"] = min(
                        global_stats["min_angle"], chunk_stats["min_angle"]
                    )
                    global_stats["max_angle"] = max(
                        global_stats["max_angle"], chunk_stats["max_angle"]
                    )
                    global_stats["sum_angles"] += chunk_stats["sum_angles"]
                    global_stats["sum_squared"] += chunk_stats["sum_squared"]
                    global_stats["count"] += chunk_stats["count"]
                    global_stats["chunks_processed"] += 1

                    # Progress reporting
                    if (
                        self.streaming_config.progress_callback
                        and global_stats["chunks_processed"] % max(1, num_chunks // 20)
                        == 0
                    ):
                        progress = global_stats["chunks_processed"] / num_chunks
                        self.streaming_config.progress_callback(
                            progress,
                            f"Processed {global_stats['chunks_processed']}/{num_chunks} chunks",
                        )

        # Compute final statistics
        mean_angle = global_stats["sum_angles"] / global_stats["count"]
        variance = (global_stats["sum_squared"] / global_stats["count"]) - mean_angle**2
        std_angle = np.sqrt(max(0, variance))

        angle_range = global_stats["max_angle"] - global_stats["min_angle"]

        analysis = {
            "n_angles": n_angles,
            "chunks_processed": global_stats["chunks_processed"],
            "angle_range_rad": angle_range,
            "angle_range_deg": np.degrees(angle_range),
            "mean_angle": mean_angle,
            "std_angle": std_angle,
            "processing_method": "parallel_streaming",
        }

        # Mode suggestion for very large datasets
        if angle_range >= np.pi * 1.2:  # > 216 degrees
            mode_suggestion = "laminar_flow"
            confidence = 0.8
            reasoning = ["Very wide angle range from parallel streaming analysis"]
        elif angle_range >= np.pi / 2:  # >= 90 degrees
            mode_suggestion = "static_anisotropic"
            confidence = 0.75
            reasoning = ["Wide angle range from parallel streaming analysis"]
        else:
            mode_suggestion = "static_isotropic"
            confidence = 0.7
            reasoning = ["Limited angle range from parallel streaming analysis"]

        analysis["reasoning"] = reasoning

        return {
            "mode_suggestion": mode_suggestion,
            "confidence": confidence,
            "analysis": analysis,
        }

    def _sequential_streaming_analysis(
        self, phi_angles: np.ndarray, detailed_analysis: bool
    ) -> Dict[str, Any]:
        """
        Sequential streaming analysis for memory-constrained environments.

        Args:
            phi_angles: Phi angle array
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            Analysis results
        """
        n_angles = len(phi_angles)
        chunk_size = self.streaming_config.chunk_size

        logger.debug(
            f"Sequential streaming analysis: {n_angles:,} angles, chunk size {chunk_size:,}"
        )

        # Online statistics computation
        count = 0
        mean = 0.0
        m2 = 0.0  # For variance calculation
        min_angle = float("inf")
        max_angle = float("-inf")
        chunks_processed = 0

        # Process chunks sequentially
        for i in range(0, n_angles, chunk_size):
            end_idx = min(i + chunk_size, n_angles)
            chunk = phi_angles[i:end_idx]
            chunk_rad = self._ensure_radians(chunk)

            # Update running statistics using Welford's online algorithm
            for angle in chunk_rad:
                count += 1
                delta = angle - mean
                mean += delta / count
                delta2 = angle - mean
                m2 += delta * delta2

                min_angle = min(min_angle, angle)
                max_angle = max(max_angle, angle)

            chunks_processed += 1

            # Memory management and progress reporting
            if chunks_processed % 100 == 0:
                gc.collect()

                if self.streaming_config.progress_callback:
                    progress = (i + len(chunk)) / n_angles
                    self.streaming_config.progress_callback(
                        progress, f"Processed {i + len(chunk):,} / {n_angles:,} angles"
                    )

        # Final statistics
        std_angle = np.sqrt(m2 / count) if count > 1 else 0.0
        angle_range = max_angle - min_angle

        analysis = {
            "n_angles": n_angles,
            "chunks_processed": chunks_processed,
            "angle_range_rad": angle_range,
            "angle_range_deg": np.degrees(angle_range),
            "mean_angle": mean,
            "std_angle": std_angle,
            "processing_method": "sequential_streaming",
        }

        # Mode suggestion
        if angle_range >= np.pi:  # >= 180 degrees
            mode_suggestion = "laminar_flow"
            confidence = 0.8
            reasoning = ["Wide angle range from sequential streaming analysis"]
        elif angle_range >= np.pi / 3:  # >= 60 degrees
            mode_suggestion = "static_anisotropic"
            confidence = 0.75
            reasoning = ["Moderate angle range from sequential streaming analysis"]
        else:
            mode_suggestion = "static_isotropic"
            confidence = 0.7
            reasoning = ["Limited angle range from sequential streaming analysis"]

        analysis["reasoning"] = reasoning

        return {
            "mode_suggestion": mode_suggestion,
            "confidence": confidence,
            "analysis": analysis,
        }

    def _analyze_data_structure_streaming(
        self, data_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze data structure with memory-efficient streaming.

        Args:
            data_dict: Data dictionary

        Returns:
            Analysis results
        """
        analysis = {
            "suggested_mode": "static_isotropic",
            "confidence": 0.3,
            "reasoning": ["Default fallback due to missing phi angle data"],
        }

        # Check correlation data size
        c2_data = data_dict.get("c2_exp")
        if c2_data is not None:
            data_size = self._safe_get_size(c2_data)

            if data_size > self.large_dataset_thresholds["very_large"]:
                analysis["suggested_mode"] = "laminar_flow"
                analysis["confidence"] = 0.6
                analysis["reasoning"] = [
                    f"Very large dataset ({data_size:,} points) suggests complex analysis appropriate"
                ]
            elif data_size > self.large_dataset_thresholds["large"]:
                analysis["suggested_mode"] = "static_anisotropic"
                analysis["confidence"] = 0.5
                analysis["reasoning"] = [
                    f"Large dataset ({data_size:,} points) suggests anisotropic analysis"
                ]
            else:
                analysis["confidence"] = 0.4
                analysis["reasoning"] = [
                    f"Medium dataset ({data_size:,} points), using isotropic default"
                ]

        return analysis

    def _compute_detailed_angle_statistics(self, phi_rad: np.ndarray) -> Dict[str, Any]:
        """
        Compute detailed angle statistics for small arrays.

        Args:
            phi_rad: Phi angles in radians

        Returns:
            Detailed statistics
        """
        if len(phi_rad) < 2:
            return {}

        # Angular statistics
        sorted_angles = np.sort(phi_rad)
        angle_differences = np.diff(sorted_angles)

        stats = {
            "angle_spacing_mean": np.mean(angle_differences),
            "angle_spacing_std": np.std(angle_differences),
            "angle_spacing_uniformity": (
                1.0 - (np.std(angle_differences) / np.mean(angle_differences))
                if np.mean(angle_differences) > 0
                else 0.0
            ),
            "angle_coverage_fraction": (np.max(phi_rad) - np.min(phi_rad))
            / (2 * np.pi),
        }

        # Add percentiles for distribution analysis
        if len(phi_rad) >= 10:
            stats.update(
                {
                    "angle_percentiles": {
                        "10th": np.percentile(phi_rad, 10),
                        "25th": np.percentile(phi_rad, 25),
                        "50th": np.percentile(phi_rad, 50),
                        "75th": np.percentile(phi_rad, 75),
                        "90th": np.percentile(phi_rad, 90),
                    }
                }
            )

        return stats

    def _validate_mode_compatibility(
        self, mode: str, data_dict: Dict[str, Any], config: Dict[str, Any]
    ) -> bool:
        """
        Validate mode compatibility with available data and configuration.

        Args:
            mode: Proposed mode
            data_dict: Data dictionary
            config: Configuration dictionary

        Returns:
            True if mode is compatible
        """
        try:
            # Use inherited compatibility analysis
            compatibility = self._analyze_comprehensive_compatibility(
                mode, config, data_dict
            )
            return compatibility.is_compatible and compatibility.confidence >= 0.5
        except Exception as e:
            logger.debug(f"Mode compatibility validation failed: {e}")
            return False

    def _generate_array_hash(self, array: np.ndarray) -> str:
        """
        Generate hash for array caching (memory efficient).

        Args:
            array: NumPy array

        Returns:
            Hash string
        """
        # For very large arrays, sample for hashing
        if array.size > 100000:
            # Sample every nth element
            step = max(1, array.size // 10000)
            sample = array[::step]
        else:
            sample = array

        # Create hash from sample statistics
        import hashlib

        hash_input = (
            f"{sample.size}:{np.min(sample)}:{np.max(sample)}:{np.mean(sample)}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    @staticmethod
    def _safe_get_size(data: Any) -> int:
        """Safely get size of data object."""
        if data is None:
            return 0

        try:
            if hasattr(data, "size"):
                return data.size
            elif hasattr(data, "__len__"):
                return len(data)
            else:
                return 1
        except (TypeError, AttributeError):
            return 0
