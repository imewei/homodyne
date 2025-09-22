"""
Progress Tracking and Reporting System
======================================

Real-time progress indicators for optimization methods with
completion time estimates and performance metrics.
"""

import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import sys

# Try to import tqdm with fallback
try:
    from tqdm import tqdm, trange
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Simple fallback progress bar
    class tqdm:
        def __init__(self, total=None, desc=None, disable=False, **kwargs):
            self.total = total
            self.desc = desc
            self.disable = disable
            self.n = 0
            self.start_time = time.time()
            if not disable and desc:
                print(f"{desc}...")

        def update(self, n=1):
            if not self.disable:
                self.n += n
                if self.total:
                    pct = (self.n / self.total) * 100
                    elapsed = time.time() - self.start_time
                    if self.n > 0:
                        eta = (elapsed / self.n) * (self.total - self.n)
                        sys.stdout.write(f"\r{self.desc}: {pct:.1f}% [{self.n}/{self.total}] ETA: {eta:.1f}s")
                        sys.stdout.flush()

        def close(self):
            if not self.disable:
                print()  # New line after progress

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    trange = lambda *args, **kwargs: tqdm(range(*args), **kwargs)


@dataclass
class OptimizationProgress:
    """Track optimization progress with performance metrics."""

    method: str
    total_iterations: int
    dataset_size: int
    start_time: float = field(default_factory=time.time)
    current_iteration: int = 0
    best_loss: Optional[float] = None
    loss_history: list = field(default_factory=list)
    convergence_checks: list = field(default_factory=list)

    # Performance metrics
    iterations_per_second: float = 0.0
    estimated_time_remaining: float = 0.0
    memory_usage_mb: float = 0.0

    # Progress bar
    pbar: Optional[tqdm] = None

    def __post_init__(self):
        """Initialize progress bar."""
        desc = f"{self.method} Optimization"

        # Add dataset size info
        if self.dataset_size > 1_000_000:
            desc += f" [{self.dataset_size/1e6:.1f}M points]"
        elif self.dataset_size > 1000:
            desc += f" [{self.dataset_size/1000:.1f}K points]"
        else:
            desc += f" [{self.dataset_size} points]"

        self.pbar = tqdm(
            total=self.total_iterations,
            desc=desc,
            unit="iter",
            dynamic_ncols=True,
            disable=not TQDM_AVAILABLE
        )

    def update(self, loss: Optional[float] = None, **metrics):
        """Update progress with current metrics."""
        self.current_iteration += 1
        elapsed = time.time() - self.start_time

        # Update loss tracking
        if loss is not None:
            self.loss_history.append(loss)
            if self.best_loss is None or loss < self.best_loss:
                self.best_loss = loss

        # Calculate performance metrics
        if self.current_iteration > 0:
            self.iterations_per_second = self.current_iteration / elapsed
            remaining_iters = self.total_iterations - self.current_iteration
            self.estimated_time_remaining = remaining_iters / self.iterations_per_second

        # Update progress bar
        if self.pbar:
            postfix = {}

            # Add loss info
            if loss is not None:
                postfix['loss'] = f"{loss:.4e}"
                if self.best_loss is not None:
                    postfix['best'] = f"{self.best_loss:.4e}"

            # Add performance metrics
            postfix['it/s'] = f"{self.iterations_per_second:.1f}"

            # Add custom metrics
            for key, value in metrics.items():
                if isinstance(value, float):
                    postfix[key] = f"{value:.4f}"
                else:
                    postfix[key] = str(value)

            self.pbar.set_postfix(postfix)
            self.pbar.update(1)

    def check_convergence(self, tolerance: float = 1e-6, window: int = 10) -> bool:
        """Check if optimization has converged."""
        if len(self.loss_history) < window:
            return False

        recent_losses = self.loss_history[-window:]
        loss_variance = max(recent_losses) - min(recent_losses)

        converged = loss_variance < tolerance
        self.convergence_checks.append({
            'iteration': self.current_iteration,
            'variance': loss_variance,
            'converged': converged
        })

        return converged

    def close(self):
        """Close progress bar and print summary."""
        if self.pbar:
            self.pbar.close()

        # Print summary
        elapsed = time.time() - self.start_time
        print(f"\n{self.method} Optimization Summary:")
        print(f"  Total iterations: {self.current_iteration}/{self.total_iterations}")
        print(f"  Time elapsed: {elapsed:.1f}s")
        print(f"  Average speed: {self.iterations_per_second:.1f} it/s")

        if self.best_loss is not None:
            print(f"  Best loss: {self.best_loss:.6e}")

            # Check if converged
            if self.convergence_checks:
                last_check = self.convergence_checks[-1]
                if last_check['converged']:
                    print(f"  ✓ Converged at iteration {last_check['iteration']}")
                else:
                    print(f"  ⚠ Not converged (variance: {last_check['variance']:.2e})")


class BatchProgressTracker:
    """Track progress for batch processing of large datasets."""

    def __init__(self, total_batches: int, batch_size: int, desc: str = "Processing"):
        self.total_batches = total_batches
        self.batch_size = batch_size
        self.total_items = total_batches * batch_size

        self.pbar = tqdm(
            total=self.total_items,
            desc=desc,
            unit="samples",
            unit_scale=True,
            dynamic_ncols=True
        )

        self.batch_times = []
        self.start_time = time.time()

    def update_batch(self, batch_idx: int, metrics: Optional[Dict[str, Any]] = None):
        """Update progress for completed batch."""
        batch_time = time.time()
        if self.batch_times:
            batch_duration = batch_time - self.batch_times[-1]
        else:
            batch_duration = batch_time - self.start_time

        self.batch_times.append(batch_time)

        # Calculate statistics
        avg_batch_time = (batch_time - self.start_time) / len(self.batch_times)
        remaining_batches = self.total_batches - batch_idx - 1
        eta = remaining_batches * avg_batch_time

        # Update progress
        postfix = {
            'batch': f"{batch_idx + 1}/{self.total_batches}",
            'batch_time': f"{batch_duration:.2f}s",
            'eta': f"{eta:.0f}s"
        }

        if metrics:
            postfix.update(metrics)

        self.pbar.set_postfix(postfix)
        self.pbar.update(self.batch_size)

    def close(self):
        """Close progress bar."""
        self.pbar.close()

        elapsed = time.time() - self.start_time
        print(f"Batch processing complete: {elapsed:.1f}s total")


def create_progress_callback(progress_tracker: OptimizationProgress) -> Callable:
    """Create a callback function for optimization libraries."""

    def callback(iteration: int, loss: float, **kwargs):
        """Progress callback for optimization."""
        progress_tracker.update(loss=loss, **kwargs)

        # Check for early stopping
        if progress_tracker.check_convergence():
            return True  # Signal early stopping

        return False

    return callback


# Convenience functions
def track_optimization(method: str, total_iterations: int, dataset_size: int) -> OptimizationProgress:
    """Create optimization progress tracker."""
    return OptimizationProgress(
        method=method,
        total_iterations=total_iterations,
        dataset_size=dataset_size
    )


def track_batches(total_batches: int, batch_size: int, desc: str = "Processing") -> BatchProgressTracker:
    """Create batch processing progress tracker."""
    return BatchProgressTracker(
        total_batches=total_batches,
        batch_size=batch_size,
        desc=desc
    )