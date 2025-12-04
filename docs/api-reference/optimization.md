# Optimization API Reference

**Version:** 3.0 (NLSQ API Alignment) **Module:** `homodyne.optimization` **Date:**
October 2025

______________________________________________________________________

## Table of Contents

1. [NLSQWrapper](#nlsqwrapper)
1. [DatasetSizeStrategy](#datasetsizestrategy)
1. [CheckpointManager](#checkpointmanager)
1. [BatchStatistics](#batchstatistics)
1. [NumericalValidator](#numericalvalidator)
1. [RecoveryStrategyApplicator](#recoverystrategyapplicator)
1. [Data Structures](#data-structures)
1. [Exceptions](#exceptions)

______________________________________________________________________

## NLSQWrapper

**Module:** `homodyne.optimization.nlsq.wrapper`

Main optimization interface providing unified access to all NLSQ strategies (STANDARD,
LARGE, CHUNKED, STREAMING).

### Class Definition

```python
class NLSQWrapper:
    """Unified interface for NLSQ optimization with automatic strategy selection.

    This wrapper provides:
    - Automatic strategy selection based on dataset size
    - Unified API across all NLSQ strategies
    - Error recovery and fault tolerance
    - Checkpoint/resume for streaming mode
    - Batch statistics tracking
    - Fast mode for production

    Attributes
    ----------
    enable_large_dataset : bool
        Enable LARGE/CHUNKED/STREAMING strategies
    enable_recovery : bool
        Enable automatic error recovery
    enable_numerical_validation : bool
        Validate for NaN/Inf at critical points
    max_retries : int
        Maximum retry attempts per optimization
    fast_mode : bool
        Disable non-essential validation for speed
    """
```

### Constructor

```python
def __init__(
    self,
    enable_large_dataset: bool = True,
    enable_recovery: bool = True,
    enable_numerical_validation: bool = True,
    max_retries: int = 2,
    fast_mode: bool = False,
) -> None:
    """Initialize NLSQWrapper.

    Parameters
    ----------
    enable_large_dataset : bool, optional
        Enable LARGE/CHUNKED/STREAMING strategies, by default True
    enable_recovery : bool, optional
        Enable automatic error recovery, by default True
    enable_numerical_validation : bool, optional
        Validate numerics (NaN/Inf), by default True
    max_retries : int, optional
        Maximum retry attempts, by default 2
    fast_mode : bool, optional
        Disable validation for <1% overhead, by default False

    Examples
    --------
    >>> # Production mode: fast, minimal overhead
    >>> wrapper = NLSQWrapper(fast_mode=True)

    >>> # Development mode: full validation
    >>> wrapper = NLSQWrapper(
    ...     enable_numerical_validation=True,
    ...     enable_recovery=True,
    ... )
    """
```

### Methods

#### fit()

Primary optimization method with automatic strategy selection.

```python
def fit(
    self,
    model_func: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    method: str = 'trf',
    config: dict | None = None,
) -> OptimizationResult:
    """Fit model to data using optimal NLSQ strategy.

    Automatically selects strategy based on dataset size:
    - < 1M points: STANDARD (curve_fit)
    - 1M-10M: LARGE (curve_fit_large)
    - 10M-100M: CHUNKED (curve_fit_large with progress)
    - > 100M: STREAMING (StreamingOptimizer)

    Parameters
    ----------
    model_func : callable
        Model function: f(xdata, *params) -> ydata_predicted
    xdata : np.ndarray
        Independent variable data (1D array of indices or values)
    ydata : np.ndarray
        Dependent variable data (1D array to fit)
    p0 : np.ndarray
        Initial parameter guess
    bounds : tuple of np.ndarray
        Parameter bounds: (lower_bounds, upper_bounds)
    method : str, optional
        Optimization method: 'trf' or 'lm', by default 'trf'
    config : dict, optional
        Configuration dict for checkpoint/streaming settings

    Returns
    -------
    OptimizationResult
        Optimization result with parameters, uncertainties, diagnostics

    Raises
    ------
    NLSQOptimizationError
        If optimization fails after all retry attempts
    NLSQNumericalError
        If unrecoverable numerical issues detected

    Examples
    --------
    >>> import numpy as np
    >>> wrapper = NLSQWrapper()
    >>>
    >>> # Define model
    >>> def model(x, a, b):
    ...     return a * x + b
    >>>
    >>> # Generate data
    >>> xdata = np.arange(1000)
    >>> ydata = 2.5 * xdata + 1.0 + np.random.randn(1000) * 0.1
    >>>
    >>> # Fit
    >>> result = wrapper.fit(
    ...     model_func=model,
    ...     xdata=xdata,
    ...     ydata=ydata,
    ...     p0=np.array([1.0, 0.0]),
    ...     bounds=(np.array([0.0, -10.0]), np.array([10.0, 10.0])),
    ... )
    >>>
    >>> print(f"Parameters: {result.parameters}")
    >>> print(f"Strategy: {result.strategy}")
    """
```

#### \_handle_nlsq_result()

Internal method for normalizing NLSQ return types.

```python
def _handle_nlsq_result(
    self,
    result: Any,
    strategy: OptimizationStrategy,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Normalize NLSQ result to consistent (popt, pcov, info) format.

    Handles four result formats:
    1. Tuple (popt, pcov) - from curve_fit_large
    2. Tuple (popt, pcov, info) - from curve_fit
    3. CurveFitResult object - from curve_fit
    4. Dict - from StreamingOptimizer

    Parameters
    ----------
    result : any
        Raw result from NLSQ function
    strategy : OptimizationStrategy
        Strategy that produced the result

    Returns
    -------
    popt : np.ndarray
        Optimal parameters
    pcov : np.ndarray
        Covariance matrix
    info : dict
        Additional information (iterations, convergence, etc.)

    Notes
    -----
    This method handles NLSQ API inconsistencies transparently.
    """
```

______________________________________________________________________

## DatasetSizeStrategy

**Module:** `homodyne.optimization.strategy`

Strategy selector for choosing optimal NLSQ method based on dataset size.

### Class Definition

```python
class DatasetSizeStrategy:
    """Select optimization strategy based on dataset size and memory.

    Strategy Thresholds:
    - < 1M points: STANDARD (curve_fit)
    - 1M-10M: LARGE (curve_fit_large)
    - 10M-100M: CHUNKED (curve_fit_large with progress)
    - > 100M: STREAMING (StreamingOptimizer)

    Memory-based adjustment enabled by default to prevent OOM errors.
    """
```

### Methods

#### select_strategy()

```python
def select_strategy(
    self,
    n_points: int,
    n_parameters: int,
    strategy_override: str | None = None,
    check_memory: bool = True,
    memory_limit_gb: float | None = None,
) -> OptimizationStrategy:
    """Select optimal optimization strategy.

    Parameters
    ----------
    n_points : int
        Number of data points
    n_parameters : int
        Number of model parameters
    strategy_override : str, optional
        Force specific strategy: 'standard', 'large', 'chunked', 'streaming'
    check_memory : bool, optional
        Adjust strategy based on available memory, by default True
    memory_limit_gb : float, optional
        Custom memory limit (GB), overrides system detection

    Returns
    -------
    OptimizationStrategy
        Selected strategy (STANDARD, LARGE, CHUNKED, or STREAMING)

    Examples
    --------
    >>> selector = DatasetSizeStrategy()
    >>>
    >>> # Automatic selection
    >>> strategy = selector.select_strategy(
    ...     n_points=50_000_000,
    ...     n_parameters=5,
    ... )
    >>> print(strategy)  # OptimizationStrategy.CHUNKED
    >>>
    >>> # Manual override
    >>> strategy = selector.select_strategy(
    ...     n_points=1_000,
    ...     n_parameters=5,
    ...     strategy_override='streaming',
    ... )
    >>> print(strategy)  # OptimizationStrategy.STREAMING
    """
```

#### build_streaming_config()

**New in v3.0** - Build optimized streaming configuration.

```python
def build_streaming_config(
    self,
    n_points: int,
    n_parameters: int,
    checkpoint_config: dict | None = None,
    memory_limit_gb: float | None = None,
) -> dict:
    """Build optimized StreamingOptimizer configuration.

    Automatically calculates optimal batch size based on available memory
    and Jacobian memory requirements.

    Parameters
    ----------
    n_points : int
        Total number of data points
    n_parameters : int
        Number of model parameters
    checkpoint_config : dict, optional
        Checkpoint settings to merge:
        - enable_checkpoints: bool
        - checkpoint_dir: str
        - checkpoint_frequency: int
        - resume_from_checkpoint: bool
        - keep_last_checkpoints: int
    memory_limit_gb : float, optional
        Custom memory limit (GB), overrides auto-detection

    Returns
    -------
    dict
        Streaming configuration with keys:
        - batch_size: int (optimized based on memory)
        - max_epochs: int
        - enable_checkpoints: bool
        - checkpoint_dir: str
        - checkpoint_frequency: int
        - resume_from_checkpoint: bool
        - keep_last_checkpoints: int
        - enable_fault_tolerance: bool
        - validate_numerics: bool
        - min_success_rate: float
        - max_retries_per_batch: int

    Examples
    --------
    >>> selector = DatasetSizeStrategy()
    >>>
    >>> # Build config with automatic batch sizing
    >>> config = selector.build_streaming_config(
    ...     n_points=200_000_000,
    ...     n_parameters=9,
    ...     checkpoint_config={
    ...         'enable_checkpoints': True,
    ...         'checkpoint_dir': './checkpoints',
    ...     },
    ... )
    >>>
    >>> print(f"Optimal batch size: {config['batch_size']}")
    >>> # Typical: 50,000 for 8 GB RAM, 9 parameters

    Notes
    -----
    Batch size calculation:
    - Target: 10% of available memory
    - Accounts for data + Jacobian (n_points × n_parameters)
    - Bounded: 1,000 to 100,000 points
    - Rounded to nearest 1,000 for clean numbers

    Typical batch sizes:
    - 1 GB available → 10,000 points
    - 8 GB available → 50,000 points
    - 32 GB available → 100,000 points (capped)
    """
```

______________________________________________________________________

## CheckpointManager

**Module:** `homodyne.optimization.checkpoint_manager`

Manage checkpoint save/load for fault-tolerant streaming optimization.

### Class Definition

```python
class CheckpointManager:
    """Manage HDF5-based checkpoints for streaming optimization.

    Features:
    - HDF5 format with gzip compression
    - SHA256 checksum validation
    - Automatic cleanup of old checkpoints
    - Version compatibility checking
    - Target save time < 2 seconds

    Attributes
    ----------
    checkpoint_dir : Path
        Directory for checkpoint files
    checkpoint_frequency : int
        Save every N batches
    keep_last_n : int
        Keep last N checkpoints (default: 3)
    enable_compression : bool
        Use HDF5 gzip compression (default: True)
    """
```

### Constructor

```python
def __init__(
    self,
    checkpoint_dir: str | Path,
    checkpoint_frequency: int = 10,
    keep_last_n: int = 3,
    enable_compression: bool = True,
):
    """Initialize checkpoint manager.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory for checkpoint files (created if doesn't exist)
    checkpoint_frequency : int, optional
        Save checkpoint every N batches, by default 10
    keep_last_n : int, optional
        Keep last N checkpoints, by default 3
    enable_compression : bool, optional
        Use HDF5 gzip compression, by default True
    """
```

### Methods

#### save_checkpoint()

```python
def save_checkpoint(
    self,
    batch_idx: int,
    parameters: np.ndarray,
    optimizer_state: dict,
    loss: float,
    metadata: dict | None = None,
) -> Path:
    """Save checkpoint to HDF5 file.

    Target save time: < 2 seconds (warns if exceeded).

    Parameters
    ----------
    batch_idx : int
        Current batch index
    parameters : np.ndarray
        Current parameter values
    optimizer_state : dict
        Optimizer internal state
    loss : float
        Current loss value
    metadata : dict, optional
        Additional metadata (batch statistics, recovery actions, etc.)

    Returns
    -------
    Path
        Path to saved checkpoint file

    Raises
    ------
    NLSQCheckpointError
        If checkpoint save fails

    Examples
    --------
    >>> manager = CheckpointManager("./checkpoints")
    >>> checkpoint_path = manager.save_checkpoint(
    ...     batch_idx=10,
    ...     parameters=np.array([1.0, 2.0, 3.0]),
    ...     optimizer_state={'iteration': 42},
    ...     loss=0.123,
    ...     metadata={'success_rate': 0.95},
    ... )
    >>> print(checkpoint_path)
    # ./checkpoints/homodyne_state_batch_0010.h5
    """
```

#### load_checkpoint()

```python
def load_checkpoint(self, checkpoint_path: Path) -> dict:
    """Load and validate checkpoint from HDF5 file.

    Performs checksum validation for integrity.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file

    Returns
    -------
    dict
        Checkpoint data with keys:
        - batch_idx: int
        - parameters: np.ndarray
        - optimizer_state: dict
        - loss: float
        - metadata: dict
        - version: str (homodyne version)
        - timestamp: float (Unix timestamp)

    Raises
    ------
    NLSQCheckpointError
        If checkpoint is corrupted, invalid, or missing

    Examples
    --------
    >>> data = manager.load_checkpoint(checkpoint_path)
    >>> print(f"Resume from batch {data['batch_idx']}")
    >>> params = data['parameters']
    >>> optimizer_state = data['optimizer_state']
    """
```

#### find_latest_checkpoint()

```python
def find_latest_checkpoint(self) -> Path | None:
    """Find most recent valid checkpoint.

    Searches checkpoint directory and returns the checkpoint with
    the highest batch index that passes validation.

    Returns
    -------
    Path or None
        Path to latest valid checkpoint, or None if none exist

    Examples
    --------
    >>> latest = manager.find_latest_checkpoint()
    >>> if latest:
    ...     data = manager.load_checkpoint(latest)
    ...     print(f"Resuming from batch {data['batch_idx']}")
    ... else:
    ...     print("No checkpoints found, starting fresh")
    """
```

#### cleanup_old_checkpoints()

```python
def cleanup_old_checkpoints(self) -> list[Path]:
    """Remove old checkpoints, keeping last N.

    Automatically called after saving new checkpoint to manage disk space.

    Returns
    -------
    list of Path
        Paths of deleted checkpoint files

    Examples
    --------
    >>> # Cleanup is automatic, but can be called manually
    >>> deleted = manager.cleanup_old_checkpoints()
    >>> print(f"Deleted {len(deleted)} old checkpoints")
    """
```

#### validate_checkpoint()

```python
def validate_checkpoint(self, checkpoint_path: Path) -> bool:
    """Validate checkpoint integrity.

    Checks file exists, has required fields, and passes checksum validation.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> if manager.validate_checkpoint(checkpoint_path):
    ...     print("Checkpoint is valid")
    ... else:
    ...     print("Checkpoint is corrupted")
    """
```

______________________________________________________________________

## BatchStatistics

**Module:** `homodyne.optimization.batch_statistics`

Circular buffer for tracking batch-level statistics during streaming optimization.

### Class Definition

```python
class BatchStatistics:
    """Track batch-level optimization statistics with circular buffer.

    Maintains statistics for the most recent N batches (default 100)
    without unbounded memory growth.

    Attributes
    ----------
    buffer : deque
        Circular buffer (max_size most recent batches)
    total_batches : int
        Total batches processed (all time)
    total_successes : int
        Total successful batches (all time)
    total_failures : int
        Total failed batches (all time)
    error_counts : dict
        Error type distribution (all time)
    """
```

### Constructor

```python
def __init__(self, max_size: int = 100):
    """Initialize batch statistics tracker.

    Parameters
    ----------
    max_size : int, optional
        Maximum batches in circular buffer, by default 100
    """
```

### Methods

#### record_batch()

```python
def record_batch(
    self,
    batch_idx: int,
    success: bool,
    loss: float,
    iterations: int,
    recovery_actions: list[str],
    error_type: str | None = None,
) -> None:
    """Record statistics for a single batch.

    Parameters
    ----------
    batch_idx : int
        Batch index (0-indexed)
    success : bool
        Whether batch optimization succeeded
    loss : float
        Final loss value for this batch
    iterations : int
        Number of iterations performed
    recovery_actions : list of str
        Recovery actions applied (if any)
    error_type : str, optional
        Type of error encountered if failed

    Examples
    --------
    >>> stats = BatchStatistics()
    >>> stats.record_batch(
    ...     batch_idx=0,
    ...     success=True,
    ...     loss=0.123,
    ...     iterations=50,
    ...     recovery_actions=[],
    ... )
    """
```

#### get_success_rate()

```python
def get_success_rate(self) -> float:
    """Calculate success rate from recent batches in buffer.

    Returns
    -------
    float
        Success rate (0.0 to 1.0)

    Examples
    --------
    >>> rate = stats.get_success_rate()
    >>> print(f"Success rate: {rate:.1%}")  # e.g., "Success rate: 93.5%"
    """
```

#### get_statistics()

```python
def get_statistics(self) -> dict[str, Any]:
    """Return comprehensive statistics dictionary.

    Returns
    -------
    dict
        Statistics with keys:
        - total_batches: Total processed (all time)
        - total_successes: Total successful (all time)
        - total_failures: Total failed (all time)
        - success_rate: Success rate from recent batches
        - average_loss: Avg loss from successful batches
        - average_iterations: Avg iterations per batch
        - error_distribution: Error type counts
        - recent_batches: List of recent batch records

    Examples
    --------
    >>> stats_dict = stats.get_statistics()
    >>> print(f"Total batches: {stats_dict['total_batches']}")
    >>> print(f"Success rate: {stats_dict['success_rate']:.1%}")
    >>> print(f"Error distribution: {stats_dict['error_distribution']}")
    """
```

______________________________________________________________________

## NumericalValidator

**Module:** `homodyne.optimization.numerical_validation`

Validate numerical stability at three critical points during optimization.

### Class Definition

```python
class NumericalValidator:
    """Validate numerical values at critical optimization points.

    Validation Points:
    1. After gradient computation (detect gradient overflow)
    2. After parameter update (detect parameter divergence)
    3. After loss calculation (detect loss NaN/Inf)

    Can be disabled via fast_mode for <1% performance overhead.

    Attributes
    ----------
    enable_validation : bool
        Whether to perform validation
    bounds : tuple of np.ndarray or None
        Parameter bounds for violation checking
    """
```

### Constructor

```python
def __init__(
    self,
    enable_validation: bool = True,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
):
    """Initialize numerical validator.

    Parameters
    ----------
    enable_validation : bool, optional
        Whether to perform validation, by default True
    bounds : tuple of np.ndarray, optional
        Parameter bounds (lower, upper) for checking
    """
```

### Methods

#### validate_gradients()

```python
def validate_gradients(self, gradients: Any) -> None:
    """Validate gradients for NaN/Inf after Jacobian computation.

    Validation Point 1: Gradients can become non-finite due to
    overflow in model function or ill-conditioned Jacobian.

    Parameters
    ----------
    gradients : array-like
        Gradient values to validate

    Raises
    ------
    NLSQNumericalError
        If gradients contain NaN or Inf values

    Examples
    --------
    >>> validator = NumericalValidator()
    >>> try:
    ...     validator.validate_gradients(gradients)
    ... except NLSQNumericalError as e:
    ...     print(f"Gradient error: {e.detection_point}")
    ...     # Apply recovery strategy
    """
```

#### validate_parameters()

```python
def validate_parameters(
    self,
    parameters: Any,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Validate parameters for NaN/Inf and bounds violations.

    Validation Point 2: Parameters can become non-finite after
    update steps, especially with aggressive step sizes.

    Parameters
    ----------
    parameters : array-like
        Parameter values to validate
    bounds : tuple of np.ndarray, optional
        Parameter bounds, overrides instance bounds

    Raises
    ------
    NLSQNumericalError
        If parameters contain NaN/Inf or violate bounds

    Examples
    --------
    >>> validator.validate_parameters(
    ...     params,
    ...     bounds=(lower_bounds, upper_bounds),
    ... )
    """
```

#### validate_loss()

```python
def validate_loss(self, loss_value: Any) -> None:
    """Validate loss value for NaN/Inf after loss computation.

    Validation Point 3: Loss can become non-finite due to
    overflow in residual computation or invalid parameters.

    Parameters
    ----------
    loss_value : scalar
        Loss value to validate

    Raises
    ------
    NLSQNumericalError
        If loss is NaN or Inf

    Examples
    --------
    >>> try:
    ...     validator.validate_loss(loss)
    ... except NLSQNumericalError as e:
    ...     print(f"Loss validation failed: {e}")
    """
```

______________________________________________________________________

## RecoveryStrategyApplicator

**Module:** `homodyne.optimization.recovery_strategies`

Apply error-specific recovery strategies when optimization fails.

### Class Definition

```python
class RecoveryStrategyApplicator:
    """Apply recovery strategies for optimization failures.

    Error-Specific Strategies:
    - NLSQConvergenceError: perturb parameters, increase iterations, relax tolerance
    - NLSQNumericalError: reduce step size, tighten bounds, rescale data

    Strategies are applied in prioritized order until max retries exhausted.

    Attributes
    ----------
    max_retries : int
        Maximum retry attempts per batch
    """
```

### Constructor

```python
def __init__(self, max_retries: int = 2):
    """Initialize recovery strategy applicator.

    Parameters
    ----------
    max_retries : int, optional
        Maximum retry attempts, by default 2
    """
```

### Methods

#### get_recovery_strategy()

```python
def get_recovery_strategy(
    self,
    error: Exception,
    params: np.ndarray,
    attempt: int,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[str, np.ndarray] | None:
    """Get recovery strategy for the given error and attempt.

    Parameters
    ----------
    error : Exception
        Exception that was raised
    params : np.ndarray
        Current parameter values
    attempt : int
        Retry attempt number (0-indexed)
    bounds : tuple of np.ndarray, optional
        Parameter bounds

    Returns
    -------
    tuple of (str, np.ndarray) or None
        (strategy_name, modified_params) if available, else None

    Examples
    --------
    >>> applicator = RecoveryStrategyApplicator(max_retries=2)
    >>> error = NLSQConvergenceError("Failed to converge")
    >>>
    >>> # First attempt
    >>> strategy, new_params = applicator.get_recovery_strategy(
    ...     error, params, attempt=0
    ... )
    >>> print(strategy)  # "perturb_parameters"
    >>>
    >>> # Second attempt
    >>> strategy, new_params = applicator.get_recovery_strategy(
    ...     error, params, attempt=1
    ... )
    >>> print(strategy)  # "increase_iterations"
    >>>
    >>> # No more strategies
    >>> result = applicator.get_recovery_strategy(
    ...     error, params, attempt=2
    ... )
    >>> print(result)  # None
    """
```

______________________________________________________________________

## Data Structures

### OptimizationResult

```python
@dataclass
class OptimizationResult:
    """Result from NLSQ optimization.

    Attributes
    ----------
    parameters : np.ndarray
        Optimal parameter values
    covariance : np.ndarray
        Covariance matrix
    uncertainties : np.ndarray
        Parameter uncertainties (std deviations)
    chi_squared : float
        Chi-squared goodness of fit
    reduced_chi_squared : float
        Reduced chi-squared (chi^2 / DOF)
    convergence_status : str
        Convergence status message
    iterations : int
        Number of iterations performed
    device_info : dict
        Device information (CPU/GPU)
    recovery_actions : list[str]
        Recovery actions applied during optimization
    strategy : str
        Strategy used: 'STANDARD', 'LARGE', 'CHUNKED', or 'STREAMING'
    streaming_diagnostics : dict or None
        Streaming-specific diagnostics (if STREAMING mode)
    """
```

### StreamingDiagnostics

Included in `OptimizationResult.streaming_diagnostics` for STREAMING mode:

```python
{
    'batch_success_rate': float,          # 0.0 to 1.0
    'failed_batch_indices': list[int],    # Indices of failed batches
    'error_type_distribution': dict,      # Error type counts
    'average_iterations_per_batch': float,
    'total_batches_processed': int,
    # ... additional metadata
}
```

### OptimizationStrategy

```python
class OptimizationStrategy(Enum):
    """Optimization strategy enum."""
    STANDARD = "standard"    # < 1M points
    LARGE = "large"          # 1M-10M points
    CHUNKED = "chunked"      # 10M-100M points
    STREAMING = "streaming"  # > 100M points
```

______________________________________________________________________

## Exceptions

### Exception Hierarchy

```python
NLSQOptimizationError           # Base exception
├── NLSQConvergenceError        # Convergence failures
├── NLSQNumericalError          # NaN/Inf/bounds violations
└── NLSQCheckpointError         # Checkpoint save/load failures
```

### NLSQOptimizationError

```python
class NLSQOptimizationError(Exception):
    """Base exception for NLSQ optimization errors.

    Attributes
    ----------
    message : str
        Error message
    error_context : dict
        Additional context for debugging
    """
```

### NLSQConvergenceError

```python
class NLSQConvergenceError(NLSQOptimizationError):
    """Raised when optimization fails to converge.

    Recovery strategies:
    1. Perturb parameters (5% random noise)
    2. Increase max iterations (1.5x)
    3. Relax convergence tolerance (10x)
    """
```

### NLSQNumericalError

```python
class NLSQNumericalError(NLSQOptimizationError):
    """Raised when numerical issues detected (NaN/Inf).

    Attributes
    ----------
    detection_point : str
        Where error was detected: 'gradient', 'parameter', or 'loss'
    invalid_values : list
        Sample of invalid values for debugging

    Recovery strategies:
    1. Reduce step size (0.5x)
    2. Tighten parameter bounds (0.9x range)
    3. Rescale data (normalize to [0, 1])
    """
```

### NLSQCheckpointError

```python
class NLSQCheckpointError(NLSQOptimizationError):
    """Raised when checkpoint save/load fails.

    Common causes:
    - Corrupted HDF5 file
    - Checksum mismatch
    - Missing required fields
    - Disk full or permissions error
    """
```

______________________________________________________________________

## Performance Characteristics

### Strategy Overhead

| Strategy | Memory Scaling | Time Overhead | Recommended For |
|----------|----------------|---------------|-----------------| | STANDARD | Linear | 0%
(baseline) | < 1M points | | LARGE | Linear | ~10% | 1M-10M points | | CHUNKED | Linear
| ~15% | 10M-100M points | | STREAMING | **Constant** | ~25% | > 100M points |

### Fault Tolerance Overhead

| Mode | Overhead | Features | |------|----------|----------| | No validation | 0% | No
error detection | | Numerical validation | ~0.5% | NaN/Inf detection | | Full fault
tolerance | < 5% | Validation + recovery | | Fast mode | < 1% | Minimal checks |

### Memory Requirements

**STREAMING mode (constant memory):**

- **Batch size:** 10% of available RAM
- **Typical usage:** 1-2 GB regardless of dataset size
- **Coefficient of variation:** < 20% across batches

**Other modes (linear scaling):**

- **STANDARD:** ~2.5× data size
- **LARGE:** ~3.0× data size (memory optimized)
- **CHUNKED:** ~3.5× data size (with progress tracking)

______________________________________________________________________

## Migration Notes

### Breaking Changes from v2.0

1. **Removed:** `performance.subsampling` configuration section
1. **Changed:** StreamingOptimizer now required for > 100M points
1. **Added:** `streaming_diagnostics` field in OptimizationResult
1. **Added:** `fast_mode` parameter to NLSQWrapper

See `/docs/migration/v2_to_v3_migration.md` for complete migration guide.

______________________________________________________________________

## References

- **NLSQ Documentation:** https://nlsq.readthedocs.io/en/latest/
- **Usage Guide:** `/docs/guides/streaming_optimizer_usage.md`
- **Migration Guide:** `/docs/migration/v2_to_v3_migration.md`
- **Performance Tuning:** `/docs/guides/performance_tuning.md`

______________________________________________________________________

**Last Updated:** October 22, 2025 **Homodyne Version:** 3.0+ **NLSQ Version:** 0.1.5+
