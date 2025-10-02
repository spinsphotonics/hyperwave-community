"""API client configuration and utilities for Hyperwave GPU services.

This module provides API configuration and helper functions for encoding/decoding
data transmitted to/from the Hyperwave API.

Main functions:
    configure_api: Set API credentials and endpoint
    encode_array: Encode numpy arrays for API transmission
    decode_array: Decode arrays from API responses

Environment Variables:
    HYPERWAVE_API_KEY: API authentication key
    HYPERWAVE_API_URL: API endpoint URL (optional, defaults to production)
"""

import os
import base64
import io
from typing import Dict, Any, Tuple, Optional

import numpy as np
import jax.numpy as jnp
import requests


# Global API configuration
_API_CONFIG = {
    'api_key': None,
    'api_url': 'https://api.hyperwave.com'  # Update with production URL
}


def configure_api(api_key: Optional[str] = None, api_url: Optional[str] = None):
    """Configure API credentials and endpoint.

    Args:
        api_key: API authentication key. If None, uses HYPERWAVE_API_KEY environment variable.
        api_url: API endpoint URL. If None, uses HYPERWAVE_API_URL environment variable
            or defaults to production endpoint.

    Raises:
        ValueError: If no API key is provided and HYPERWAVE_API_KEY is not set.

    Example:
        >>> import hyperwave_community as hwc
        >>> hwc.configure_api(api_key='your-key-here')
        >>> # Or use environment variable
        >>> import os
        >>> os.environ['HYPERWAVE_API_KEY'] = 'your-key-here'
    """
    global _API_CONFIG

    if api_key is not None:
        _API_CONFIG['api_key'] = api_key
    elif 'HYPERWAVE_API_KEY' in os.environ:
        _API_CONFIG['api_key'] = os.environ['HYPERWAVE_API_KEY']

    if api_url is not None:
        _API_CONFIG['api_url'] = api_url
    elif 'HYPERWAVE_API_URL' in os.environ:
        _API_CONFIG['api_url'] = os.environ['HYPERWAVE_API_URL']

    if _API_CONFIG['api_key'] is None:
        raise ValueError(
            "API key not provided. Set HYPERWAVE_API_KEY environment variable "
            "or call configure_api(api_key='your-key') first."
        )


def _get_api_config() -> Dict[str, str]:
    """Get current API configuration.

    Returns:
        Dictionary with 'api_key' and 'api_url' keys.

    Raises:
        RuntimeError: If API is not configured.
    """
    if _API_CONFIG['api_key'] is None:
        # Try to load from environment
        if 'HYPERWAVE_API_KEY' in os.environ:
            _API_CONFIG['api_key'] = os.environ['HYPERWAVE_API_KEY']
        else:
            raise RuntimeError(
                "API not configured. Call configure_api() or set HYPERWAVE_API_KEY "
                "environment variable first."
            )

    return _API_CONFIG


def encode_array(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string for API transmission.

    Args:
        arr: Numpy array to encode.

    Returns:
        Base64-encoded string representation of array.
    """
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_array(b64_str: str) -> np.ndarray:
    """Decode base64 string to numpy array.

    Args:
        b64_str: Base64-encoded string.

    Returns:
        Decoded numpy array.
    """
    buffer = io.BytesIO(base64.b64decode(b64_str))
    return np.load(buffer)


def simulate(
    structure,
    source_field: jnp.ndarray,
    source_offset: Tuple[int, int, int],
    freq_band: Tuple[float, float, int],
    monitors,
    mode_info: Optional[Dict] = None,
    max_steps: int = 10000,
    check_every_n: int = 1000,
    source_ramp_periods: float = 5.0,
    add_absorption: bool = True,
    absorption_widths: Tuple[int, int, int] = (70, 35, 17),
    absorption_coeff: float = 4.89e-3,
    gpu_type: str = "H100",
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run FDTD simulation on GPU via API.

    Submits structure, source, and monitors to remote GPU server for FDTD simulation.
    Returns field data at monitor locations, convergence information, and power analysis.

    Args:
        structure: Structure object with permittivity and conductivity.
        source_field: Source field array, shape (num_freqs, 6, x, y, z).
        source_offset: Corner position (x, y, z) for source placement.
        freq_band: Frequency specification as (min, max, num_points).
        monitors: MonitorSet object containing field monitors.
        mode_info: Optional dictionary with mode information (beta, field, error).
        max_steps: Maximum FDTD time steps.
        check_every_n: Convergence check interval (in time steps).
        source_ramp_periods: Number of periods for source turn-on.
        add_absorption: If True, add PML absorption boundaries on GPU.
        absorption_widths: PML widths as (x_width, y_width, z_width) in pixels.
        absorption_coeff: PML absorption coefficient.
        gpu_type: GPU type to use (H100, A100, A10G, L4).
        api_key: API key (overrides configured key).
        api_url: API URL (overrides configured URL).

    Returns:
        Dictionary containing:
            - monitor_data: Dict mapping monitor names to field arrays
            - monitor_names: Dict mapping names to indices
            - convergence: Tuple of (steps, errors)
            - performance: Grid-points × steps per second
            - powers: Dict of power values per monitor
            - transmissions: Dict of transmission values
            - sim_time: GPU simulation time in seconds
            - gpu_type: GPU type used

    Raises:
        RuntimeError: If API call fails.
        ConnectionError: If cannot connect to API endpoint.

    Note:
        Typical execution time:
        - Cold start (first request): ~60-70 seconds (includes container startup)
        - Warm (subsequent): ~25-30 seconds (GPU compute only)
        - Large structures may take longer

    Example:
        >>> import hyperwave_community as hwc
        >>> # Setup
        >>> structure = hwc.create_structure(layers=[...])
        >>> source, offset, _ = hwc.create_mode_source(...)
        >>> monitors = hwc.MonitorSet()
        >>> monitors.add_monitors_at_position(structure, axis='x', position=100)
        >>>
        >>> # Run simulation
        >>> results = hwc.simulate(
        ...     structure=structure,
        ...     source_field=source,
        ...     source_offset=offset,
        ...     freq_band=(2*jnp.pi/1.6, 2*jnp.pi/1.5, 2),
        ...     monitors=monitors
        ... )
        >>> print(f"Transmission: {results['transmissions']['transmission']}")
    """
    # Get API configuration
    config = _get_api_config()
    if api_key is not None:
        config['api_key'] = api_key
    if api_url is not None:
        config['api_url'] = api_url

    # Extract structure recipe
    structure_recipe = structure.extract_recipe()

    # Encode source field
    source_field_b64 = encode_array(np.array(source_field))

    # Serialize monitors
    monitors_serialized = {}
    monitor_tuple = monitors.to_tuple()
    for i, (name, monitor) in enumerate(zip(monitors.list_monitors(), monitor_tuple[0])):
        monitors_serialized[name] = {
            'shape': list(monitor.shape),
            'offset': list(monitor.offset),
            'index': i
        }

    # Prepare mode_info
    mode_info_serialized = None
    if mode_info is not None:
        mode_info_serialized = {
            k: v.tolist() if isinstance(v, (np.ndarray, jnp.ndarray)) else v
            for k, v in mode_info.items()
        }

    # Prepare request
    request_data = {
        "structure_recipe": structure_recipe,
        "source_field_b64": source_field_b64,
        "source_field_shape": list(source_field.shape),
        "source_offset": list(source_offset),
        "freq_band": list(freq_band),
        "monitors": monitors_serialized,
        "mode_info": mode_info_serialized,
        "max_steps": max_steps,
        "check_every_n": check_every_n,
        "source_ramp_periods": source_ramp_periods,
        "add_absorption": add_absorption,
        "absorption_widths": list(absorption_widths),
        "absorption_coeff": absorption_coeff,
        "gpu_type": gpu_type
    }

    # Send request
    try:
        response = requests.post(
            f"{config['api_url']}/simulate",
            json=request_data,
            headers={"Authorization": f"Bearer {config['api_key']}"},
            timeout=600  # 10 minute timeout
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        results = response.json()

        # Decode monitor data
        monitor_data = {}
        for name, b64_str in results['monitor_data_b64'].items():
            monitor_data[name] = decode_array(b64_str)

        # Decode powers and transmissions
        powers = {name: decode_array(b64_str) for name, b64_str in results['powers'].items()}
        transmissions = {name: decode_array(b64_str) for name, b64_str in results['transmissions'].items()}

        # Decode convergence
        conv_steps = decode_array(results['convergence_steps'])
        conv_errors = {k: decode_array(v) for k, v in results['convergence_errors'].items()}

        return {
            'monitor_data': monitor_data,
            'monitor_names': results['monitor_names'],
            'convergence': (conv_steps, list(conv_errors.values())),
            'performance': results['performance'],
            'powers': powers,
            'transmissions': transmissions,
            'sim_time': results['sim_time'],
            'gpu_type': results['gpu_type']
        }

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not connect to API at {config['api_url']}. "
            "Check your network connection and API URL."
        )
    except Exception as e:
        raise RuntimeError(f"API request failed: {str(e)}")


def generate_gaussian_source(
    structure_shape: Tuple[int, int, int, int],
    conductivity_boundary: jnp.ndarray,
    freq_band: Tuple[float, float, int],
    source_z_pos: int,
    polarization: str = 'x',
    max_steps: int = 5000,
    check_every_n: int = 1000,
    gpu_type: str = "H100",
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate unidirectional Gaussian source on GPU via API.

    Creates a truly unidirectional Gaussian beam using the wave equation error
    method on remote GPU. This prevents reflection artifacts.

    Args:
        structure_shape: Simulation domain shape as (3, Lx, Ly, Lz).
        conductivity_boundary: Absorption boundary array, shape (Lx, Ly, Lz).
        freq_band: Frequency specification as (min, max, num_points).
        source_z_pos: Z-position for source injection (in pixels).
        polarization: Polarization direction, 'x' or 'y'.
        max_steps: Maximum FDTD steps for source generation.
        check_every_n: Convergence check interval.
        gpu_type: GPU type to use (H100, A100, A10G, L4).
        api_key: API key (overrides configured key).
        api_url: API URL (overrides configured URL).

    Returns:
        Dictionary containing:
            - source_field: Generated source field array
            - source_field_shape: Shape of source field
            - source_power: Power per frequency
            - source_position: Source offset position
            - total_time: Total generation time
            - fdtd_time: FDTD simulation time
            - gpu_type: GPU type used

    Raises:
        RuntimeError: If API call fails.
        ConnectionError: If cannot connect to API endpoint.

    Note:
        Generation takes approximately 20-30 seconds on H100 GPU.

    Example:
        >>> import hyperwave_community as hwc
        >>> # Create absorption mask
        >>> abs_mask = hwc.create_absorption_mask(
        ...     shape=(500, 500, 200),
        ...     absorption_widths=(90, 90, 90)
        ... )
        >>>
        >>> # Generate source
        >>> result = hwc.generate_gaussian_source(
        ...     structure_shape=(3, 500, 500, 200),
        ...     conductivity_boundary=abs_mask,
        ...     freq_band=(2*jnp.pi/0.55, 2*jnp.pi/0.55, 1),
        ...     source_z_pos=60,
        ...     polarization='x'
        ... )
        >>> source_field = result['source_field']
    """
    # Get API configuration
    config = _get_api_config()
    if api_key is not None:
        config['api_key'] = api_key
    if api_url is not None:
        config['api_url'] = api_url

    # Encode conductivity boundary
    conductivity_b64 = encode_array(np.array(conductivity_boundary))

    # Prepare request
    request_data = {
        "structure_shape": list(structure_shape),
        "conductivity_boundary_b64": conductivity_b64,
        "freq_band": list(freq_band),
        "source_z_pos": source_z_pos,
        "polarization": polarization,
        "max_steps": max_steps,
        "check_every_n": check_every_n,
        "gpu_type": gpu_type
    }

    # Send request
    try:
        response = requests.post(
            f"{config['api_url']}/generate_gaussian_source",
            json=request_data,
            headers={"Authorization": f"Bearer {config['api_key']}"},
            timeout=600  # 10 minute timeout
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        results = response.json()

        # Decode source field
        source_field = decode_array(results['source_field_b64'])

        return {
            'source_field': source_field,
            'source_field_shape': results['source_field_shape'],
            'source_power': results['source_power'],
            'source_position': tuple(results['source_position']),
            'total_time': results['total_time'],
            'fdtd_time': results['fdtd_time'],
            'gpu_type': results['gpu_type']
        }

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not connect to API at {config['api_url']}. "
            "Check your network connection and API URL."
        )
    except Exception as e:
        raise RuntimeError(f"API request failed: {str(e)}")
