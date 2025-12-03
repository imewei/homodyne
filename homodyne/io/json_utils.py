"""JSON utility functions for homodyne I/O operations.

This module provides helper functions for JSON serialization of numpy arrays
and other complex objects.
"""

from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Recursively convert numpy arrays and special types to JSON-safe types.

    Parameters
    ----------
    value : Any
        Value to convert (can be nested dict, list, numpy array, etc.)

    Returns
    -------
    Any
        JSON-serializable version of the input

    Examples
    --------
    >>> json_safe(np.array([1, 2, 3]))
    [1, 2, 3]
    >>> json_safe({"arr": np.array([1.0, 2.0]), "val": np.float64(3.14)})
    {'arr': [1.0, 2.0], 'val': 3.14}
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, (np.bool_,)):
        return bool(value)
    elif hasattr(value, "tolist"):
        return value.tolist()
    else:
        return value


def json_serializer(obj: Any) -> Any:
    """JSON serializer for numpy arrays and other objects.

    Use as the `default` argument to json.dump/dumps.

    Parameters
    ----------
    obj : Any
        Object to serialize

    Returns
    -------
    Any
        JSON-serializable version of the object

    Raises
    ------
    TypeError
        If object cannot be serialized (will be converted to string)

    Examples
    --------
    >>> import json
    >>> json.dumps({"arr": np.array([1, 2, 3])}, default=json_serializer)
    '{"arr": [1, 2, 3]}'
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    else:
        return str(obj)
