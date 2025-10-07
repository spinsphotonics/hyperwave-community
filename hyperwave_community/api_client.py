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
    'api_url': 'https://hyperwave-cloud.onrender.com'
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


def generate_gaussian_source(
    structure_shape: Tuple[int, int, int],
    conductivity_boundary: jnp.ndarray,
    freq_band: Tuple[float, float, int],
    source_z_pos: int,
    polarization: str = 'x',
    simulation_steps: int = 5000,
    check_every_n: int = 1000,
    gpu_type: str = "H100",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate unidirectional Gaussian source on GPU via API.

    Creates a truly unidirectional Gaussian beam using the wave equation error
    method on remote GPU. This prevents reflection artifacts.

    Args:
        structure_shape: Simulation domain shape as (Lx, Ly, Lz).
        conductivity_boundary: Absorption boundary array, shape (Lx, Ly, Lz).
        freq_band: Frequency specification as (min, max, num_points).
        source_z_pos: Z-position for source injection (in pixels).
        polarization: Polarization direction, 'x' or 'y'.
        simulation_steps: Number of FDTD time steps for source generation.
            The simulation will converge to a relatively low error at around this step count.
        check_every_n: Convergence check interval.
        gpu_type: GPU type to use. Options: B200, H200, H100, A100-80GB, A100-40GB, L40S, L4, A10G, T4.
            Default: H100.
        api_key: API key (overrides configured key).

    Returns:
        Dictionary containing:
            - source_field: Generated source field array
            - source_field_shape: Shape of source field
            - source_power: Power per frequency
            - source_position: Source offset position
            - total_time: Total generation time
            - fdtd_time: FDTD simulation time
            - gpu_type: GPU type used
            - simulation_id: Unique ID for this simulation
            - execution_time_seconds: Total execution time
            - computation_time_seconds: GPU computation time

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
        ...     structure_shape=(500, 500, 200),
        ...     conductivity_boundary=abs_mask,
        ...     freq_band=(2*jnp.pi/0.55, 2*jnp.pi/0.55, 1),
        ...     source_z_pos=60,
        ...     polarization='x',
        ...     api_key='your-api-key'
        ... )
        >>> source_field = result['source_field']
    """
    # Check for API key
    if not api_key:
        print("API key required to proceed.")
        print("Sign up for free at spinsphotonics.com to get your API key.")
        return None

    API_URL = "https://hyperwave-cloud.onrender.com"

    # Encode conductivity boundary
    conductivity_b64 = encode_array(np.array(conductivity_boundary))

    # Prepare request - add the '3' back for API compatibility
    # Convert all values to native Python types for JSON serialization (handles JAX arrays)
    request_data = {
        "structure_shape": [3] + [int(x) for x in structure_shape],  # API expects (3, Lx, Ly, Lz)
        "conductivity_boundary_b64": conductivity_b64,
        "freq_band": [float(x) for x in freq_band],
        "source_z_pos": int(source_z_pos),
        "polarization": polarization,
        "simulation_steps": int(simulation_steps),
        "check_every_n": int(check_every_n),
        "gpu_type": gpu_type
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    # Send request
    try:
        response = requests.post(
            f"{API_URL}/generate_gaussian_source",
            json=request_data,
            headers=headers,
            timeout=600  # 10 minute timeout
        )

        response.raise_for_status()  # raises HTTPError if status != 200

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
            'gpu_type': results['gpu_type'],
            'simulation_id': results.get('simulation_id'),
            'execution_time_seconds': results.get('execution_time_seconds'),
            'computation_time_seconds': results.get('computation_time_seconds')
        }

    except requests.exceptions.HTTPError as e:
        # Access the response from the exception object
        if e.response is not None:
            status_code = e.response.status_code
            response_text = e.response.text

            if status_code == 401:
                print("No API key detected in request.")
                print("Sign up for free at spinsphotonics.com to get your API key.")
                return None
            elif status_code == 403:
                print("Provided API key is invalid.")
                print("Please verify your API key in your dashboard at spinsphotonics.com/dashboard")
                return None
            elif status_code == 402:
                # Try to extract current balance from response if available
                try:
                    error_data = e.response.json()
                    current_balance = error_data.get("current_balance", 0)
                    balance_msg = f"Current balance: {current_balance:.4f} credits"
                except:
                    balance_msg = ""

                print("Insufficient credits for Gaussian source generation.")
                print("Minimum required: 0.1 credits")
                if balance_msg:
                    print(balance_msg)
                print("Add credits to your account at spinsphotonics.com/billing")
                return None
            elif status_code == 502:
                print("Service temporarily unavailable.")
                print("Our servers are experiencing high load. Please retry in a few moments.")
                return None
            else:
                print(f"Unexpected error (Code: {status_code})")
                print("Please try again or contact support if the issue persists.")
                return None
        else:
            print("Communication error.")
            print("Unable to process your request at this time. Please try again later.")
            return None

    except requests.exceptions.Timeout:
        print("Request timeout.")
        print("The source generation server is taking longer than expected. Please try again.")
        return None

    except requests.exceptions.ConnectionError as e:
        print("Connection failed.")
        print("Unable to reach source generation servers. Please check your network connection and try again.")
        return None

    except requests.exceptions.RequestException as e:
        print("Communication error.")
        print("Unable to process your request at this time. Please try again later.")
        return None

    except ValueError as e:
        print("Invalid server response.")
        print("Received malformed data from server. Our team has been notified.")
        return None
