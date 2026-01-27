"""API client configuration and utilities for Hyperwave GPU services.

This module provides API configuration and helper functions for encoding/decoding
data transmitted to/from the Hyperwave API.

Main functions:
    configure_api: Set API credentials and endpoint
    encode_array: Encode numpy arrays for API transmission
    decode_array: Decode arrays from API responses

GPU Simulation Functions:
    simulate: Run FDTD simulation (in simulate.py)
    early_stopping_simulate: Run FDTD with early stopping
    generate_gaussian_source: Generate Gaussian source field

Recipe Builder Functions (CPU, no credits):
    build_recipe: Build structure recipe from gdsfactory component
    build_monitors: Build monitors from port info
    solve_mode_source: Solve waveguide mode source
    compute_freq_band: Compute frequency band from wavelengths
    get_default_absorber_params: Get default absorber parameters
    prepare_simulation_inputs: Prepare ALL simulation inputs in one call

Utility Functions:
    estimate_cost: Estimate simulation cost before running

Environment Variables:
    HYPERWAVE_API_KEY: API authentication key
    HYPERWAVE_API_URL: API endpoint URL (optional, defaults to production)
"""

import os
import base64
import io
from typing import Dict, Any, Tuple, Optional, List

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


def _handle_api_error(e: requests.exceptions.HTTPError, operation: str) -> None:
    """Handle common API errors with user-friendly messages."""
    if e.response is not None:
        status_code = e.response.status_code

        if status_code == 401:
            print("No API key detected in request.")
            print("Sign up for free at spinsphotonics.com to get your API key.")
        elif status_code == 403:
            print("Provided API key is invalid.")
            print("Please verify your API key in your dashboard at spinsphotonics.com/dashboard")
        elif status_code == 402:
            try:
                error_data = e.response.json()
                current_balance = error_data.get("current_balance", 0)
                balance_msg = f"Current balance: {current_balance:.4f} credits"
            except:
                balance_msg = ""

            print(f"Insufficient credits for {operation}.")
            if balance_msg:
                print(balance_msg)
            print("Add credits to your account at spinsphotonics.com/billing")
        elif status_code == 429:
            print("Too many concurrent simulations.")
            print("Please wait for existing simulations to complete.")
        elif status_code == 502:
            print("Service temporarily unavailable.")
            print("Our servers are experiencing high load. Please retry in a few moments.")
        else:
            print(f"Unexpected error (Code: {status_code})")
            print("Please try again or contact support if the issue persists.")
    else:
        print("Communication error.")
        print("Unable to process your request at this time. Please try again later.")


# =============================================================================
# ACCOUNT INFO
# =============================================================================

def get_account_info() -> Optional[Dict[str, Any]]:
    """Verify API key and get account information including credit balance.

    Returns:
        Dictionary with:
            - valid: Whether the API key is valid
            - credits_balance: Current credit balance
            - credits_balance_usd: Credit balance in USD ($10 = 1 credit)
        Returns None if request fails.

    Example:
        >>> import hyperwave_community as hwc
        >>> hwc.configure_api(api_key='your-key-here')
        >>> info = hwc.get_account_info()
        >>> if info:
        ...     print(f"API key valid: {info['valid']}")
        ...     print(f"Credits: {info['credits_balance']:.4f} (${info['credits_balance_usd']:.2f})")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    try:
        response = requests.post(
            f"{API_URL}/account_info",
            params={"api_key": API_KEY},
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            print("Invalid API key.")
            print("Please verify your API key in your dashboard at spinsphotonics.com/dashboard")
        else:
            print(f"Error getting account info: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting account info: {e}")
        return None


# =============================================================================
# COST ESTIMATION
# =============================================================================

def estimate_cost(
    grid_points: Optional[int] = None,
    structure_shape: Optional[Tuple[int, int, int, int]] = None,
    max_steps: int = 10000,
    gpu_type: str = "H100",
    simulation_type: str = "fdtd_simulation",
) -> Optional[Dict[str, Any]]:
    """Estimate simulation cost before running (no auth required).

    Args:
        grid_points: Total grid points (Lx * Ly * Lz). Either this or structure_shape required.
        structure_shape: Structure shape (3, Lx, Ly, Lz). Alternative to grid_points.
        max_steps: Maximum FDTD steps.
        gpu_type: GPU type (B200, H200, H100, A100-80GB, A100-40GB, L40S, A10G, T4).
        simulation_type: Type of simulation (fdtd_simulation, gaussian_source).

    Returns:
        Dictionary with estimated_seconds, estimated_credits, estimated_cost_usd, gpu_type, grid_points, note.
        Returns None if request fails.

    Example:
        >>> estimate = hwc.estimate_cost(
        ...     structure_shape=(3, 500, 300, 150),
        ...     max_steps=20000,
        ...     gpu_type="H100"
        ... )
        >>> print(f"Estimated cost: ${estimate['estimated_cost_usd']:.2f} ({estimate['estimated_credits']:.4f} credits)")
    """
    API_URL = _API_CONFIG['api_url']

    request_data = {
        "max_steps": max_steps,
        "gpu_type": gpu_type,
        "simulation_type": simulation_type
    }

    if grid_points is not None:
        request_data["grid_points"] = grid_points
    elif structure_shape is not None:
        request_data["structure_shape"] = list(structure_shape)
    else:
        print("Either grid_points or structure_shape must be provided.")
        return None

    try:
        response = requests.post(
            f"{API_URL}/estimate_cost",
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error estimating cost: {e}")
        return None


# =============================================================================
# EARLY STOPPING SIMULATION
# =============================================================================

def early_stopping_simulate(
    structure,
    source_field: jnp.ndarray,
    source_offset: Tuple[int, int, int],
    freq_band: Tuple[float, float, int],
    monitors,
    mode_info: Optional[Dict] = None,
    max_steps: int = 200000,
    check_every_n: int = 5000,
    source_ramp_periods: float = 5.0,
    add_absorption: bool = True,
    absorption_widths: Tuple[int, int, int] = (70, 35, 17),
    absorption_coeff: float = 4.89e-3,
    relative_threshold: float = 0.01,
    absolute_threshold: float = 1e-8,
    significant_power_threshold: float = 1e-6,
    min_stable_checks: int = 3,
    gpu_type: str = "H100",
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run FDTD simulation with early stopping on GPU via API.

    Early stopping monitors power at all ports and stops when power stabilizes,
    significantly reducing simulation time for well-converging structures.

    Args:
        structure: Structure object with permittivity and conductivity.
        source_field: Source field array, shape (num_freqs, 6, x, y, z).
        source_offset: Corner position (x, y, z) for source placement.
        freq_band: Frequency specification as (min, max, num_points).
        monitors: MonitorSet object containing field monitors.
        mode_info: Optional dictionary with mode information.
        max_steps: Maximum FDTD time steps (default higher for early stopping).
        check_every_n: Convergence check interval (in time steps).
        source_ramp_periods: Number of periods for source turn-on.
        add_absorption: If True, add PML absorption boundaries on GPU.
        absorption_widths: PML widths as (x_width, y_width, z_width) in pixels.
        absorption_coeff: PML absorption coefficient.
        relative_threshold: Relative power change threshold for convergence.
        absolute_threshold: Absolute power threshold.
        significant_power_threshold: Minimum power to be considered significant.
        min_stable_checks: Minimum stable checks before stopping.
        gpu_type: GPU type to use.
        api_key: API key (overrides configured key).

    Returns:
        Dictionary containing:
            - monitor_data: Dict mapping monitor names to field arrays
            - monitor_names: Dict mapping names to indices
            - power_history: Dict of power history per monitor
            - converged: Whether simulation converged
            - convergence_step: Step at which convergence occurred
            - performance: Grid-points × steps per second
            - sim_time: GPU simulation time in seconds
            - gpu_type: GPU type used

    Example:
        >>> results = hwc.early_stopping_simulate(
        ...     structure=structure,
        ...     source_field=source,
        ...     source_offset=offset,
        ...     freq_band=freq_band,
        ...     monitors=monitors,
        ...     max_steps=200000,
        ...     relative_threshold=0.01,
        ...     api_key='your-key'
        ... )
        >>> print(f"Converged at step {results['convergence_step']}")
    """
    # Use configured API key if not explicitly provided
    if not api_key:
        try:
            config = _get_api_config()
            api_key = config['api_key']
        except RuntimeError:
            print("API key required. Call configure_api() or pass api_key parameter.")
            return None

    API_URL = _API_CONFIG['api_url']

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
        "relative_threshold": relative_threshold,
        "absolute_threshold": absolute_threshold,
        "significant_power_threshold": significant_power_threshold,
        "min_stable_checks": min_stable_checks,
        "gpu_type": gpu_type
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/early_stopping",
            json=request_data,
            headers=headers,
            timeout=1800  # 30 minute timeout for long simulations
        )
        response.raise_for_status()
        results = response.json()

        # Decode monitor data
        monitor_data = {}
        for name, b64_str in results.get('monitor_data_b64', {}).items():
            monitor_data[name] = decode_array(b64_str)

        # Decode power history
        power_history = {}
        for name, b64_str in results.get('power_history_b64', {}).items():
            power_history[name] = decode_array(b64_str)

        return {
            'monitor_data': monitor_data,
            'monitor_names': results.get('monitor_names', {}),
            'power_history': power_history,
            'converged': results.get('converged', False),
            'convergence_step': results.get('convergence_step', 0),
            'num_checks': results.get('num_checks', 0),
            'max_power_seen': results.get('max_power_seen', 0.0),
            'performance': results.get('performance', 0.0),
            'sim_time': results.get('sim_time', 0.0),
            'gpu_type': results.get('gpu_type', gpu_type),
            'simulation_id': results.get('simulation_id'),
            'execution_time_seconds': results.get('execution_time_seconds'),
            'computation_time_seconds': results.get('computation_time_seconds')
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "early stopping simulation")
        return None
    except requests.exceptions.Timeout:
        print("Request timeout. The simulation is taking longer than expected.")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection failed. Please check your network connection.")
        return None
    except requests.exceptions.RequestException:
        print("Communication error. Please try again later.")
        return None


# =============================================================================
# GAUSSIAN SOURCE GENERATION
# =============================================================================

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
) -> Dict[str, Any]:
    """Generate unidirectional Gaussian source on GPU via API.

    NOTE: This endpoint may have interface mismatches. Use with caution.

    Args:
        structure_shape: Simulation domain shape as (3, Lx, Ly, Lz).
        conductivity_boundary: Absorption boundary array, shape (Lx, Ly, Lz).
        freq_band: Frequency specification as (min, max, num_points).
        source_z_pos: Z-position for source injection (in pixels).
        polarization: Polarization direction, 'x' or 'y'.
        max_steps: Maximum FDTD steps for source generation.
        check_every_n: Convergence check interval.
        gpu_type: GPU type to use.
        api_key: API key (overrides configured key).

    Returns:
        Dictionary containing source_field, source_power, source_position, etc.
    """
    # Use configured API key if not explicitly provided
    if not api_key:
        try:
            config = _get_api_config()
            api_key = config['api_key']
        except RuntimeError:
            print("API key required. Call configure_api() or pass api_key parameter.")
            return None

    API_URL = _API_CONFIG['api_url']

    # Encode conductivity boundary
    conductivity_b64 = encode_array(np.array(conductivity_boundary))

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

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/generate_gaussian_source",
            json=request_data,
            headers=headers,
            timeout=600
        )
        response.raise_for_status()
        results = response.json()

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
        _handle_api_error(e, "Gaussian source generation")
        return None
    except requests.exceptions.Timeout:
        print("Request timeout.")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection failed.")
        return None
    except requests.exceptions.RequestException:
        print("Communication error.")
        return None


# =============================================================================
# RECIPE BUILDER FUNCTIONS (CPU, no credits consumed)
# =============================================================================

def build_recipe(
    component_name: str,
    component_kwargs: Optional[Dict[str, Any]] = None,
    extension_length: float = 2.0,
    resolution_nm: float = 30.0,
    n_core: float = 3.48,
    n_clad: float = 1.4457,
    wg_height_um: float = 0.22,
    total_height_um: float = 4.0,
    padding: Tuple[int, int, int, int] = (100, 100, 0, 0),
    density_radius: int = 3,
    vertical_radius: float = 2.0,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build structure recipe from gdsfactory component on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        component_name: Name of gdsfactory component (e.g., "mmi2x2").
        component_kwargs: Kwargs to pass to component constructor.
        extension_length: Length to extend ports in um.
        resolution_nm: Grid resolution in nanometers.
        n_core: Core refractive index.
        n_clad: Cladding refractive index.
        wg_height_um: Waveguide height in um.
        total_height_um: Total structure height in um.
        padding: (left, right, top, bottom) padding in theta pixels.
        density_radius: Radius for density filtering.
        vertical_radius: Vertical blur radius.
        api_key: API key (overrides configured key).

    Returns:
        Dictionary containing:
            - recipe: Structure recipe for simulation
            - density_core: 2D core density array
            - density_clad: 2D clad density array
            - dimensions: (Lx, Ly, Lz) structure dimensions
            - port_info: Dict of port positions and orientations
            - layer_config: Layer thickness configuration
            - eps_values: (eps_clad, eps_core) permittivity values

    Example:
        >>> result = hwc.build_recipe(
        ...     component_name="mmi2x2",
        ...     resolution_nm=30,
        ... )
        >>> print(f"Dimensions: {result['dimensions']}")
    """
    # Use configured API key if not explicitly provided
    if not api_key:
        try:
            config = _get_api_config()
            api_key = config['api_key']
        except RuntimeError:
            print("API key required. Call configure_api() or pass api_key parameter.")
            return None

    API_URL = _API_CONFIG['api_url']

    request_data = {
        "component_name": component_name,
        "component_kwargs": component_kwargs,
        "extension_length": extension_length,
        "resolution_nm": resolution_nm,
        "n_core": n_core,
        "n_clad": n_clad,
        "wg_height_um": wg_height_um,
        "total_height_um": total_height_um,
        "padding": list(padding),
        "density_radius": density_radius,
        "vertical_radius": vertical_radius,
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/build_recipe",
            json=request_data,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        results = response.json()

        return {
            'recipe': results['recipe'],
            'density_core': decode_array(results['density_core_b64']),
            'density_clad': decode_array(results['density_clad_b64']),
            'dimensions': tuple(results['dimensions']),
            'port_info': results['port_info'],
            'layer_config': results['layer_config'],
            'eps_values': tuple(results['eps_values']),
            'resolution_um': results['resolution_um'],
            'padding': tuple(results['padding']),
            'device_info': results.get('device_info'),
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "recipe building")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error building recipe: {e}")
        return None


def build_monitors(
    port_info: Dict[str, Any],
    dimensions: Tuple[int, int, int],
    source_port: str = "o2",
    monitor_x_um: float = 0.1,
    monitor_y_um: float = 1.5,
    monitor_z_um: float = 1.5,
    resolution_um: float = 0.03,
    source_offset_cells: int = 5,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build monitors from port information on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        port_info: Dict of port info from build_recipe.
        dimensions: (Lx, Ly, Lz) from build_recipe.
        source_port: Port name for source injection (e.g., "o2").
        monitor_x_um: Monitor thickness in um.
        monitor_y_um: Monitor Y extent in um.
        monitor_z_um: Monitor Z extent in um.
        resolution_um: Grid resolution in um.
        source_offset_cells: Cells before monitor for source position.
        api_key: API key (overrides configured key).

    Returns:
        Dictionary containing monitors, monitor_names, source_port_name,
        source_position, mode_bounds.
    """
    # Use configured API key if not explicitly provided
    if not api_key:
        try:
            config = _get_api_config()
            api_key = config['api_key']
        except RuntimeError:
            print("API key required. Call configure_api() or pass api_key parameter.")
            return None

    API_URL = _API_CONFIG['api_url']

    request_data = {
        "port_info": port_info,
        "dimensions": list(dimensions),
        "source_port": source_port,
        "monitor_x_um": monitor_x_um,
        "monitor_y_um": monitor_y_um,
        "monitor_z_um": monitor_z_um,
        "resolution_um": resolution_um,
        "source_offset_cells": source_offset_cells,
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/build_monitors",
            json=request_data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "monitor building")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error building monitors: {e}")
        return None


def solve_mode_source(
    density_core: np.ndarray,
    density_clad: np.ndarray,
    source_x_position: int,
    mode_bounds: Dict[str, int],
    layer_config: Dict[str, Any],
    eps_values: Tuple[float, float],
    freq_band: Tuple[float, float, int],
    slice_half_width: int = 5,
    mode_num: int = 0,
    propagation_axis: str = "x",
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Solve for waveguide mode source field on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        density_core: 2D core density array from build_recipe.
        density_clad: 2D clad density array from build_recipe.
        source_x_position: X position for source (in structure coords).
        mode_bounds: Dict with y_min, y_max, z_min, z_max from build_monitors.
        layer_config: Layer configuration from build_recipe.
        eps_values: (eps_clad, eps_core) from build_recipe.
        freq_band: (omega_min, omega_max, n_freqs) frequency band.
        slice_half_width: Half-width of structure slice in theta pixels.
        mode_num: Which mode to solve for (0 = fundamental).
        propagation_axis: Propagation direction ("x", "-x", "y", "-y").
        api_key: API key (overrides configured key).

    Returns:
        Dictionary containing source_field, source_offset, mode_info, freq_band.
    """
    # Use configured API key if not explicitly provided
    if not api_key:
        try:
            config = _get_api_config()
            api_key = config['api_key']
        except RuntimeError:
            print("API key required. Call configure_api() or pass api_key parameter.")
            return None

    API_URL = _API_CONFIG['api_url']

    request_data = {
        "density_core_b64": encode_array(np.array(density_core)),
        "density_core_shape": list(density_core.shape),
        "density_clad_b64": encode_array(np.array(density_clad)),
        "density_clad_shape": list(density_clad.shape),
        "source_x_position": source_x_position,
        "mode_bounds": mode_bounds,
        "layer_config": layer_config,
        "eps_values": list(eps_values),
        "freq_band": list(freq_band),
        "slice_half_width": slice_half_width,
        "mode_num": mode_num,
        "propagation_axis": propagation_axis,
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/solve_mode_source",
            json=request_data,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        results = response.json()

        return {
            'source_field': decode_array(results['source_field_b64']),
            'source_offset': tuple(results['source_offset']),
            'mode_info': results.get('mode_info', {}),
            'freq_band': tuple(results['freq_band']),
            'solve_time_seconds': results.get('solve_time_seconds', 0.0),
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "mode source solving")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error solving mode source: {e}")
        return None


def compute_freq_band(
    wl_min_um: float,
    wl_max_um: float,
    n_freqs: int = 1,
    resolution_um: float = 0.03,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compute frequency band from wavelength range on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        wl_min_um: Minimum wavelength in micrometers.
        wl_max_um: Maximum wavelength in micrometers.
        n_freqs: Number of frequency points.
        resolution_um: Grid resolution in micrometers.
        api_key: API key (overrides configured key).

    Returns:
        Dictionary containing freq_band, wavelengths_um, frequencies_omega.

    Example:
        >>> result = hwc.compute_freq_band(
        ...     wl_min_um=1.5,
        ...     wl_max_um=1.6,
        ...     n_freqs=5,
        ...     api_key='your-key'
        ... )
        >>> print(f"Freq band: {result['freq_band']}")
    """
    # Use configured API key if not explicitly provided
    if not api_key:
        try:
            config = _get_api_config()
            api_key = config['api_key']
        except RuntimeError:
            print("API key required. Call configure_api() or pass api_key parameter.")
            return None

    API_URL = _API_CONFIG['api_url']

    request_data = {
        "wl_min_um": wl_min_um,
        "wl_max_um": wl_max_um,
        "n_freqs": n_freqs,
        "resolution_um": resolution_um,
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/compute_freq_band",
            json=request_data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        results = response.json()

        return {
            'freq_band': tuple(results['freq_band']),
            'wavelengths_um': results['wavelengths_um'],
            'frequencies_omega': results['frequencies_omega'],
            'resolution_um': results['resolution_um'],
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "frequency band computation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error computing freq band: {e}")
        return None


def get_default_absorber_params(
    structure_dimensions: Tuple[int, int, int],
    absorber_fraction: float = 0.1,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Get default absorber parameters based on structure size on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        structure_dimensions: (Lx, Ly, Lz) structure dimensions.
        absorber_fraction: Fraction of each dimension for absorber (default 10%).
        api_key: API key (overrides configured key).

    Returns:
        Dictionary containing absorption_widths, absorption_coeff, add_absorption.
    """
    # Use configured API key if not explicitly provided
    if not api_key:
        try:
            config = _get_api_config()
            api_key = config['api_key']
        except RuntimeError:
            print("API key required. Call configure_api() or pass api_key parameter.")
            return None

    API_URL = _API_CONFIG['api_url']

    request_data = {
        "structure_dimensions": list(structure_dimensions),
        "absorber_fraction": absorber_fraction,
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/get_default_absorber_params",
            json=request_data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        results = response.json()

        return {
            'absorption_widths': tuple(results['absorption_widths']),
            'absorption_coeff': results['absorption_coeff'],
            'add_absorption': results['add_absorption'],
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "absorber parameter computation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting absorber params: {e}")
        return None


def prepare_simulation_inputs(
    component_name: str,
    component_kwargs: Optional[Dict[str, Any]] = None,
    extension_length: float = 2.0,
    resolution_nm: float = 30.0,
    n_core: float = 3.48,
    n_clad: float = 1.45,
    wg_height_um: float = 0.22,
    clad_top_um: float = 1.89,
    clad_bot_um: float = 2.0,
    padding: Tuple[int, int, int, int] = (100, 100, 0, 0),
    density_radius: int = 3,
    vertical_radius: float = 2.0,
    source_port: str = "o2",
    wl_min_um: float = 1.55,
    wl_max_um: float = 1.55,
    n_freqs: int = 1,
    monitor_x_um: float = 0.1,
    monitor_y_um: float = 1.5,
    monitor_z_um: float = 1.5,
    mode_num: int = 0,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Prepare ALL inputs needed for GPU simulation in one call on Modal CPU.

    This is a convenience function that combines build_recipe, build_monitors,
    and solve_mode_source into a single API call. Runs on CPU, no credits consumed.

    Args:
        component_name: Name of gdsfactory component.
        component_kwargs: Kwargs for component constructor.
        extension_length: Port extension length in um.
        resolution_nm: Grid resolution in nm.
        n_core: Core refractive index.
        n_clad: Cladding refractive index.
        wg_height_um: Waveguide height in um.
        clad_top_um: Top cladding thickness in um.
        clad_bot_um: Bottom cladding (BOX) thickness in um.
        padding: Theta padding.
        density_radius: Density filter radius.
        vertical_radius: Vertical blur radius.
        source_port: Source port name.
        wl_min_um: Min wavelength in um.
        wl_max_um: Max wavelength in um.
        n_freqs: Number of frequencies.
        monitor_x_um: Monitor thickness.
        monitor_y_um: Monitor Y extent.
        monitor_z_um: Monitor Z extent.
        mode_num: Mode number to solve.
        api_key: API key (overrides configured key).

    Returns:
        Dictionary with ALL simulation inputs:
            - structure_recipe: Recipe for reconstruction
            - source_field: Mode source field
            - source_offset: Source offset tuple
            - freq_band: Frequency band tuple
            - monitors: Monitor list (recipe format)
            - mode_info: Mode information
            - absorber_params: Default absorber parameters
            - dimensions: Structure dimensions
            - metadata: Build metadata

    Example:
        >>> inputs = hwc.prepare_simulation_inputs(
        ...     component_name="mmi2x2",
        ...     resolution_nm=30,
        ...     wl_min_um=1.55,
        ...     api_key='your-key'
        ... )
        >>> # Then use inputs directly with simulate()
    """
    # Use configured API key if not explicitly provided
    if not api_key:
        try:
            config = _get_api_config()
            api_key = config['api_key']
        except RuntimeError:
            print("API key required. Call configure_api() or pass api_key parameter.")
            return None

    API_URL = _API_CONFIG['api_url']

    request_data = {
        "component_name": component_name,
        "component_kwargs": component_kwargs,
        "extension_length": extension_length,
        "resolution_nm": resolution_nm,
        "n_core": n_core,
        "n_clad": n_clad,
        "wg_height_um": wg_height_um,
        "clad_top_um": clad_top_um,
        "clad_bot_um": clad_bot_um,
        "padding": list(padding),
        "density_radius": density_radius,
        "vertical_radius": vertical_radius,
        "source_port": source_port,
        "wl_min_um": wl_min_um,
        "wl_max_um": wl_max_um,
        "n_freqs": n_freqs,
        "monitor_x_um": monitor_x_um,
        "monitor_y_um": monitor_y_um,
        "monitor_z_um": monitor_z_um,
        "mode_num": mode_num,
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/prepare_simulation_inputs",
            json=request_data,
            headers=headers,
            timeout=600
        )
        response.raise_for_status()
        results = response.json()

        return {
            'structure_recipe': results['structure_recipe'],
            'source_field': decode_array(results['source_field_b64']),
            'source_offset': tuple(results['source_offset']),
            'freq_band': tuple(results['freq_band']),
            'monitors': results['monitors'],
            'mode_info': results.get('mode_info', {}),
            'absorber_params': results['absorber_params'],
            'dimensions': tuple(results['dimensions']),
            'metadata': results.get('metadata', {}),
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "simulation input preparation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error preparing simulation inputs: {e}")
        return None
