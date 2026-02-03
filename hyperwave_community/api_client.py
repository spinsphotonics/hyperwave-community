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
from typing import Dict, Any, Tuple, Optional, Callable

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


def prepare_structure_recipe(structure_recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Convert large density patterns to base64 before sending to API.

    This reduces request size and prevents memory issues on the server.
    Arrays larger than 100x100 are converted to base64 encoding.

    Args:
        structure_recipe: Dictionary from structure.extract_recipe()

    Returns:
        Modified recipe with large density patterns encoded as base64
    """
    import copy

    # Make a deep copy to avoid modifying original
    recipe = copy.deepcopy(structure_recipe)

    if 'layers_info' in recipe:
        for i, layer in enumerate(recipe['layers_info']):
            if 'density_pattern' in layer and isinstance(layer['density_pattern'], (list, np.ndarray)):
                arr = np.array(layer['density_pattern'], dtype=np.float32)

                # Convert large arrays to base64 (>100x100 = 10,000 elements)
                if arr.size > 10000:
                    buffer = io.BytesIO()
                    np.save(buffer, arr)
                    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

                    # Add base64 version and shape
                    layer['density_pattern_b64'] = b64_str
                    layer['density_pattern_shape'] = list(arr.shape)

                    # Remove the raw array to save memory
                    del layer['density_pattern']

                    # Log the conversion
                    size_mb = len(b64_str) / (1024**2)
                    print(f"  Encoded layer {i} density: {arr.shape} → base64 ({size_mb:.2f} MB)")

    return recipe


def generate_gaussian_source(
    structure_shape: Tuple[int, int, int],
    freq_band: Tuple[float, float, int],
    source_pos: Tuple[int, int, int],
    waist_radius: float,
    x_span: float,
    y_span: float,
    absorption_widths: Tuple[int, int, int] = (70, 35, 17),
    absorption_coeff: float = 1e-4,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = 'x',
    simulation_steps: int = 5000,
    check_every_n: int = 1000,
    gpu_type: str = "H100",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate unidirectional Gaussian source on GPU via API.

    Creates a truly unidirectional Gaussian beam using the wave equation error
    method on remote GPU. Absorption boundaries are created on backend.

    Args:
        structure_shape: Simulation domain shape as (Lx, Ly, Lz).
        freq_band: Frequency specification as (min, max, num_points).
        source_pos: Full 3D source position (x, y, z) in pixels.
        waist_radius: Beam waist radius in pixels (controls beam size).
        x_span: Source extent in X direction in pixels.
        y_span: Source extent in Y direction in pixels.
        absorption_widths: Absorption boundary widths (x, y, z) in pixels.
            Backend creates conductivity boundary from these dimensions.
            Default: (70, 35, 17).
        absorption_coeff: PML absorption coefficient (conductivity strength).
            Higher values increase absorption but may cause reflections.
            Default: 1e-4.
        theta: Tilt angle in degrees for beam steering. Default: 0.0 (normal incidence).
        phi: Azimuthal angle in degrees for beam steering. Default: 0.0.
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
        >>> import jax.numpy as jnp
        >>>
        >>> # Generate Gaussian source with custom parameters
        >>> result = hwc.api_client.generate_gaussian_source(
        ...     structure_shape=(500, 500, 200),
        ...     freq_band=(2*jnp.pi*0.3/0.6, 2*jnp.pi*0.3/0.5, 10),
        ...     source_pos=(250, 250, 60),
        ...     waist_radius=10.0,
        ...     x_span=100,
        ...     y_span=100,
        ...     theta=8.0,  # 8 degree tilt for grating coupler
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

    # Prepare request - add the '3' back for API compatibility
    # Convert all values to native Python types for JSON serialization (handles JAX arrays)
    # NOTE: Send only absorption dimensions, not full array - backend creates it
    request_data = {
        "structure_shape": [3] + [int(x) for x in structure_shape],  # API expects (3, Lx, Ly, Lz)
        "freq_band": [float(x) for x in freq_band],
        "source_pos": [int(x) for x in source_pos],
        "waist_radius": float(waist_radius),
        "x_span": float(x_span),
        "y_span": float(y_span),
        "absorption_widths": [int(x) for x in absorption_widths],
        "absorption_coeff": float(absorption_coeff),
        "theta": float(theta),
        "phi": float(phi),
        "polarization": polarization,
        "max_steps": int(simulation_steps),
        "check_every_n": int(check_every_n),
        "gpu_type": gpu_type
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    # Debug: Print request details (without sensitive data)
    print(f"\n=== API Request Debug Info ===")
    print(f"Endpoint: {API_URL}/generate_gaussian_source")
    print(f"Parameters:")
    for key, value in request_data.items():
        if key == "conductivity_boundary_b64":
            print(f"  {key}: <base64 data, {len(value)} chars>")
        else:
            print(f"  {key}: {value}")
    print(f"===============================\n")

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

        # Decode source field - API returns error_source_plane_b64, not source_field_b64
        source_field = decode_array(results['error_source_plane_b64'])

        return {
            'source_field': source_field,
            'source_field_shape': results['error_source_plane_shape'],
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

            # Debug: Print response details
            print(f"\n=== API Error Debug Info ===")
            print(f"Status Code: {status_code}")
            print(f"Response: {response_text}")
            print(f"===========================\n")

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


def compute_adjoint_gradient(
    theta: np.ndarray,
    source_field: np.ndarray,
    source_offset: Tuple[int, int, int],
    freq_band: Tuple[float, float, int],
    loss_monitor_shape: Tuple[int, int, int],
    loss_monitor_offset: Tuple[int, int, int],
    design_monitor_shape: Tuple[int, int, int],
    design_monitor_offset: Tuple[int, int, int],
    structure_spec: Dict[str, Any],
    loss_fn: Optional[Callable] = None,
    mode_field: Optional[np.ndarray] = None,
    input_power: Optional[float] = None,
    mode_cross_power: Optional[float] = None,
    mode_axis: int = 0,
    power_axis: Optional[int] = None,
    power_maximize: bool = True,
    intensity_component: Optional[str] = None,
    intensity_maximize: bool = True,
    absorption_widths: Tuple[int, int, int] = (70, 35, 17),
    absorption_coeff: float = 0.00489,
    gpu_type: str = "H100",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute adjoint gradient for inverse design on GPU via API.

    Computes the gradient of a loss function with respect to design variables
    (theta) using the memory-efficient adjoint method. This enables gradient-based
    optimization of photonic devices.

    The 3-part autodiff chain:
        theta -> permittivity (via structure_spec) -> fields (FDTD) -> loss

    Loss function options (in priority order):
        1. loss_fn: Custom function (most flexible)
        2. mode_field: Mode coupling efficiency
        3. power_axis: Poynting vector power (S_from_slice)
        4. intensity_component: Simple |E|^2

    Args:
        theta: Design variables (2D numpy array). Values typically in [0, 1].
        source_field: Source field array, shape (n_freq, 6, sx, sy, sz).
        source_offset: Source injection position (x, y, z) in pixels.
        freq_band: Frequency specification as (omega_min, omega_max, num_freqs).
        loss_monitor_shape: Shape of loss monitor. For power loss, use the
            full cross-section. For point intensity, use (1, 1, 1).
        loss_monitor_offset: Position where loss is computed (x, y, z).
        design_monitor_shape: Shape of design region for gradient computation.
        design_monitor_offset: Offset of design region.
        structure_spec: Structure specification dictionary with:
            - layers_info: list of layer dicts with density_radius, density_alpha,
              permittivity_values, layer_thickness, conductivity_values
            - construction_params: dict with vertical_radius
        loss_fn: Custom loss function (optional). Signature: loss_fn(loss_field) -> scalar.
            loss_field shape: (n_freq, 6, mx, my, mz). Serialized via cloudpickle.
            WARNING: Avoid closures over large arrays (causes memory leaks).
        mode_field: Target mode field for mode coupling loss (optional).
            If provided, also requires input_power and mode_cross_power.
        input_power: Input power for mode coupling loss.
        mode_cross_power: Mode cross power for mode coupling loss.
        mode_axis: Axis for mode overlap calculation (default: 0 for x-propagation).
        power_axis: Axis for Poynting vector power loss (optional).
            0=x, 1=y, 2=z. Uses S_from_slice to compute power through monitor.
        power_maximize: Whether to maximize power (default: True).
        intensity_component: Field component for intensity loss ('Ex', 'Ey', 'Ez').
        intensity_maximize: Whether to maximize intensity (default: True).
        absorption_widths: PML absorption widths (x, y, z) in pixels.
        absorption_coeff: PML absorption coefficient.
        gpu_type: GPU type to use. Options: B200, H200, H100, A100-80GB, A100-40GB, L40S, A10G, T4.
        api_key: API key (overrides configured key).

    Returns:
        Dictionary containing:
            - loss: Computed loss value
            - grad_theta: Gradient array (same shape as theta)
            - grad_min: Minimum gradient value
            - grad_max: Maximum gradient value
            - grad_time: Gradient computation time (seconds)
            - total_time: Total time including overhead
            - gpu_type: GPU type used
            - memory_reduction_pct: Memory reduction from efficient adjoint
            - simulation_id: Unique ID for this computation

    Raises:
        ValueError: If no loss specification is provided.

    Note:
        Computation takes approximately 30-120 seconds depending on structure size
        and GPU type. Uses memory-efficient adjoint method (90%+ memory reduction).

    Example 1: Custom loss function
        >>> def my_loss(loss_field):
        ...     import jax.numpy as jnp
        ...     Ez = loss_field[0, 2, :, :, :]
        ...     return -jnp.sum(jnp.abs(Ez)**2)  # Maximize total Ez intensity
        >>>
        >>> result = hwc.compute_adjoint_gradient(
        ...     theta=theta, source_field=source, ...,
        ...     loss_fn=my_loss,
        ...     api_key='your-key'
        ... )

    Example 2: Power through monitor (using S_from_slice)
        >>> result = hwc.compute_adjoint_gradient(
        ...     theta=theta, source_field=source, ...,
        ...     loss_monitor_shape=(1, 50, 50),  # Full YZ cross-section
        ...     loss_monitor_offset=(output_x, 0, 0),
        ...     power_axis=0,  # Maximize Sx (power in x-direction)
        ...     power_maximize=True,
        ...     api_key='your-key'
        ... )

    Example 3: Simple intensity
        >>> result = hwc.compute_adjoint_gradient(
        ...     theta=theta, source_field=source, ...,
        ...     loss_monitor_shape=(1, 1, 1),  # Point monitor
        ...     intensity_component='Ez',
        ...     intensity_maximize=True,
        ...     api_key='your-key'
        ... )
    """
    # Check for API key
    if not api_key:
        print("API key required to proceed.")
        print("Sign up for free at spinsphotonics.com to get your API key.")
        return None

    # Validate loss parameters - at least one must be provided
    if (loss_fn is None and mode_field is None and
        power_axis is None and intensity_component is None):
        raise ValueError(
            "Must provide at least one loss specification:\n"
            "  - loss_fn: Custom loss function\n"
            "  - mode_field: Mode coupling (with input_power, mode_cross_power)\n"
            "  - power_axis: Poynting vector power (0=x, 1=y, 2=z)\n"
            "  - intensity_component: Simple |E|^2 ('Ex', 'Ey', 'Ez')"
        )

    API_URL = "https://hyperwave-cloud.onrender.com"

    # Encode theta and source_field to base64
    theta_b64 = encode_array(np.array(theta, dtype=np.float32))
    source_field_b64 = encode_array(np.array(source_field))

    # Serialize custom loss function if provided
    loss_fn_pickle_b64 = None
    if loss_fn is not None:
        import cloudpickle
        loss_fn_bytes = cloudpickle.dumps(loss_fn)
        loss_fn_pickle_b64 = base64.b64encode(loss_fn_bytes).decode('utf-8')

    # Prepare mode coupling params if provided
    mode_coupling_params = None
    if mode_field is not None:
        if input_power is None or mode_cross_power is None:
            raise ValueError(
                "mode_field requires input_power and mode_cross_power"
            )
        mode_coupling_params = {
            'mode_field_b64': encode_array(np.array(mode_field)),
            'mode_field_shape': list(np.array(mode_field).shape),
            'input_power': float(input_power),
            'mode_cross_power': float(mode_cross_power),
            'axis': int(mode_axis)
        }

    # Prepare power params if provided
    power_params = None
    if power_axis is not None:
        power_params = {
            'axis': int(power_axis),
            'maximize': power_maximize
        }

    # Prepare intensity params if provided
    intensity_params = None
    if intensity_component is not None:
        intensity_params = {
            'component': intensity_component,
            'maximize': intensity_maximize
        }

    # Build request
    request_data = {
        "theta_b64": theta_b64,
        "theta_shape": list(np.array(theta).shape),
        "source_field_b64": source_field_b64,
        "source_field_shape": list(np.array(source_field).shape),
        "source_offset": list(source_offset),
        "freq_band": [float(x) for x in freq_band],
        "loss_monitor_shape": list(loss_monitor_shape),
        "loss_monitor_offset": list(loss_monitor_offset),
        "design_monitor_shape": list(design_monitor_shape),
        "design_monitor_offset": list(design_monitor_offset),
        "structure_spec": structure_spec,
        "mode_coupling_params": mode_coupling_params,
        "intensity_params": intensity_params,
        "power_params": power_params,
        "loss_fn_pickle_b64": loss_fn_pickle_b64,
        "add_absorption": True,
        "absorption_widths": list(absorption_widths),
        "absorption_coeff": float(absorption_coeff),
        "gpu_type": gpu_type
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    # Debug info
    print(f"\n=== Inverse Design API Request ===")
    print(f"Endpoint: {API_URL}/inverse_design")
    print(f"Theta shape: {request_data['theta_shape']}")
    print(f"Design monitor: shape={design_monitor_shape}, offset={design_monitor_offset}")
    print(f"Loss monitor: shape={loss_monitor_shape}, offset={loss_monitor_offset}")
    print(f"GPU type: {gpu_type}")
    if loss_fn_pickle_b64:
        print(f"Loss type: Custom function (cloudpickle)")
    elif mode_coupling_params:
        print(f"Loss type: Mode coupling")
    elif power_params:
        axis_names = {0: 'x', 1: 'y', 2: 'z'}
        print(f"Loss type: Power S_{axis_names.get(power_axis, power_axis)} via S_from_slice (maximize={power_maximize})")
    elif intensity_params:
        print(f"Loss type: Intensity |{intensity_component}|^2 (maximize={intensity_maximize})")
    print(f"==================================\n")

    # Send request
    try:
        response = requests.post(
            f"{API_URL}/inverse_design",
            json=request_data,
            headers=headers,
            timeout=1800  # 30 minute timeout for inverse design
        )

        response.raise_for_status()

        results = response.json()

        # Decode gradient
        grad_theta = decode_array(results['grad_theta_b64'])

        return {
            'loss': results['loss'],
            'grad_theta': grad_theta,
            'grad_min': results['grad_min'],
            'grad_max': results['grad_max'],
            'grad_time': results['grad_time'],
            'total_time': results['total_time'],
            'gpu_type': results['gpu_type'],
            'memory_reduction_pct': results['memory_reduction_pct'],
            'simulation_id': results.get('simulation_id'),
            'execution_time_seconds': results.get('execution_time_seconds'),
            'computation_time_seconds': results.get('computation_time_seconds')
        }

    except requests.exceptions.HTTPError as e:
        if e.response is not None:
            status_code = e.response.status_code
            response_text = e.response.text

            print(f"\n=== API Error ===")
            print(f"Status Code: {status_code}")
            print(f"Response: {response_text}")
            print(f"=================\n")

            if status_code == 401:
                print("No API key detected in request.")
                print("Sign up for free at spinsphotonics.com to get your API key.")
            elif status_code == 403:
                print("Provided API key is invalid.")
                print("Please verify your API key at spinsphotonics.com/dashboard")
            elif status_code == 402:
                print("Insufficient credits for inverse design computation.")
                print("Minimum required: 1.0 credits")
                print("Add credits at spinsphotonics.com/billing")
            elif status_code == 502:
                print("Service temporarily unavailable. Please retry later.")
            else:
                print(f"Unexpected error (Code: {status_code})")
        return None

    except requests.exceptions.Timeout:
        print("Request timeout.")
        print("Inverse design computation is taking longer than expected. Please try again.")
        return None

    except requests.exceptions.ConnectionError:
        print("Connection failed.")
        print("Unable to reach API servers. Please check your connection.")
        return None

    except requests.exceptions.RequestException:
        print("Communication error.")
        print("Unable to process your request. Please try again later.")
        return None

    except ValueError:
        print("Invalid server response.")
        print("Received malformed data from server.")
        return None
