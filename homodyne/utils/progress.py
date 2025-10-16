"""Minimal Progress Tracking for NLSQ Optimization
================================================

Since NumPyro has built-in progress bars for MCMC, this module only provides
progress tracking for NLSQ optimization, which doesn't have native
progress reporting capabilities.
"""

import sys
import time
from collections.abc import Callable

# Try to import tqdm with graceful fallback
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class OptimizationProgress:
    """Minimal progress tracker for NLSQ optimization."""

    def __init__(
        self,
        max_iterations: int,
        desc: str = "NLSQ Optimization",
        verbose: bool = True,
    ):
        """Initialize progress tracker.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations/evaluations expected
        desc : str
            Description for progress display
        verbose : bool
            Whether to show progress
        """
        self.max_iterations = max_iterations
        self.desc = desc
        self.verbose = verbose
        self.eval_count = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update_interval = max(1, max_iterations // 100)  # Update every 1%

        if self.verbose and TQDM_AVAILABLE:
            self.pbar = tqdm(total=max_iterations, desc=desc, unit="eval")
        else:
            self.pbar = None
            if self.verbose:
                print(f"{desc} starting (max {max_iterations} evaluations)...")

    def update(self, n: int = 1):
        """Update progress by n evaluations."""
        self.eval_count += n

        if not self.verbose:
            return

        current_time = time.time()

        if self.pbar:
            # tqdm progress bar update
            self.pbar.update(n)

            # Update postfix with timing info every update_interval
            if self.eval_count % self.update_interval == 0:
                elapsed = current_time - self.start_time
                rate = self.eval_count / elapsed if elapsed > 0 else 0
                eta = (self.max_iterations - self.eval_count) / rate if rate > 0 else 0

                self.pbar.set_postfix({"eval/s": f"{rate:.1f}", "eta": f"{eta:.0f}s"})
        else:
            # Simple text progress (no tqdm)
            if (
                self.eval_count % self.update_interval == 0
                or self.eval_count >= self.max_iterations
            ):
                elapsed = current_time - self.start_time
                percent = (self.eval_count / self.max_iterations) * 100
                rate = self.eval_count / elapsed if elapsed > 0 else 0
                eta = (self.max_iterations - self.eval_count) / rate if rate > 0 else 0

                sys.stdout.write(
                    f"\r{self.desc}: {percent:5.1f}% [{self.eval_count:6d}/{self.max_iterations}] "
                    f"| {rate:.1f} eval/s | ETA: {eta:.0f}s",
                )
                sys.stdout.flush()

    def close(self):
        """Close progress display and show final statistics."""
        if self.pbar:
            self.pbar.close()
        elif self.verbose:
            print()  # New line after progress

        if self.verbose:
            elapsed = time.time() - self.start_time
            print(f"\n{self.desc} completed:")
            print(f"  Total evaluations: {self.eval_count}")
            print(f"  Time elapsed: {elapsed:.1f}s")
            print(f"  Average rate: {self.eval_count / elapsed:.1f} eval/s")


def wrap_objective_with_progress(
    objective_fn: Callable,
    max_iterations: int,
    verbose: bool = True,
) -> tuple[Callable, OptimizationProgress]:
    """Wrap an objective function to track evaluation count.

    Parameters
    ----------
    objective_fn : Callable
        The objective function to wrap
    max_iterations : int
        Expected maximum iterations
    verbose : bool
        Whether to show progress

    Returns
    -------
    wrapped_fn : Callable
        Wrapped objective function that tracks progress
    progress : OptimizationProgress
        Progress tracker object (call .close() when done)
    """
    progress = OptimizationProgress(max_iterations, verbose=verbose)

    def wrapped_objective(*args, **kwargs):
        result = objective_fn(*args, **kwargs)
        progress.update(1)
        return result

    return wrapped_objective, progress
