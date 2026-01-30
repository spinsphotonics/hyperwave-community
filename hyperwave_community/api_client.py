"""Hyperwave API Client for GPU-accelerated FDTD photonics simulations.

Four workflow levels are available:

1. TWO-STAGE WORKFLOW (Recommended for most users):
    - prepare_simulation() - Setup simulation parameters (CPU on Modal)
    - run_simulation() - Run FDTD with pre-computed setup (GPU on Modal)

2. GRANULAR WORKFLOW (For advanced control):
    - build_recipe() - Create structure from GDSFactory component
    - build_monitors() - Create monitors from port information
    - compute_freq_band() - Convert wavelengths to frequencies
    - solve_mode_source() - Solve for waveguide mode
    - get_default_absorber_params() - Get absorber configuration

3. NEW GRANULAR WORKFLOW (For fine-grained control):
    - load_component() - Load GDSFactory component metadata
    - create_structure_recipe() - Create structure recipe from component
    - create_monitors() - Create monitors from structure recipe
    - solve_mode() - Solve waveguide mode at source port
    - run_gpu_simulation() - Run FDTD simulation on GPU
    - analyze_transmission() - Analyze transmission from results
    - get_field_slice() - Extract 2D field slice for visualization

4. ONE-SHOT WORKFLOW (For quick tests):
    - simulate() - Combines setup + simulation in one call

Utility Functions:
    - configure_api() - Set API credentials and endpoint
    - get_account_info() - Get account info and credit balance
    - estimate_cost() - Estimate simulation cost before running

Environment Variables:
    HYPERWAVE_API_KEY: API authentication key
    HYPERWAVE_API_URL: API endpoint URL (optional, defaults to production)
"""

import os
import base64
import io
import time
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import requests
from dataclasses import dataclass, field


# =============================================================================
# CONVERGENCE CONFIGURATION
# =============================================================================

@dataclass
class ConvergenceConfig:
    """Configuration for early stopping convergence behavior.

    Use this for fine-grained control over when simulations stop.
    For most users, the string presets ("quick", "default", "thorough", "full")
    are recommended instead.

    Attributes:
        check_every_n: Steps between convergence checks (default: 1000).
        relative_threshold: Relative power change threshold (default: 0.01 = 1%).
        min_stable_checks: Consecutive stable checks required (default: 3).
        min_steps: Minimum steps before checking convergence (default: 0).
        power_threshold: Ignore ports with power below this (default: 1e-6).
        monitors: List of specific monitor names to check. If None, checks all output ports.

    Example:
        >>> config = hwc.ConvergenceConfig(
        ...     check_every_n=500,
        ...     min_stable_checks=5,
        ...     min_steps=3000,
        ... )
        >>> results = hwc.run_simulation(..., convergence=config)
    """
    check_every_n: int = 1000
    relative_threshold: float = 0.01
    min_stable_checks: int = 3
    min_steps: int = 0
    power_threshold: float = 1e-6
    monitors: Optional[List[str]] = None


# Preset convergence configurations
# All presets use 1% relative threshold - differences are in check frequency and stability requirements
CONVERGENCE_PRESETS = {
    "quick": ConvergenceConfig(
        check_every_n=2000,
        relative_threshold=0.01,
        min_stable_checks=2,
        min_steps=0,
        power_threshold=1e-5,
    ),
    "default": ConvergenceConfig(
        check_every_n=1000,
        relative_threshold=0.01,
        min_stable_checks=3,
        min_steps=0,
        power_threshold=1e-6,
    ),
    "thorough": ConvergenceConfig(
        check_every_n=1000,
        relative_threshold=0.01,
        min_stable_checks=5,
        min_steps=5000,
        power_threshold=1e-7,
    ),
    "full": None,  # No early stopping - runs all steps
}


def _resolve_convergence(convergence) -> Optional[ConvergenceConfig]:
    """Resolve convergence parameter to a ConvergenceConfig or None.

    Args:
        convergence: Can be:
            - str: Preset name ("quick", "default", "thorough", "full")
            - ConvergenceConfig: Custom configuration
            - bool: True -> "default", False -> "full" (legacy support)
            - None: Uses "default"

    Returns:
        ConvergenceConfig or None (for "full"/no early stopping)
    """
    if convergence is None:
        return CONVERGENCE_PRESETS["default"]

    if isinstance(convergence, bool):
        # Legacy support: True -> default early stopping, False -> no early stopping
        return CONVERGENCE_PRESETS["default"] if convergence else None

    if isinstance(convergence, str):
        if convergence not in CONVERGENCE_PRESETS:
            valid = list(CONVERGENCE_PRESETS.keys())
            raise ValueError(f"Unknown convergence preset '{convergence}'. Valid options: {valid}")
        return CONVERGENCE_PRESETS[convergence]

    if isinstance(convergence, ConvergenceConfig):
        return convergence

    raise TypeError(f"convergence must be str, bool, ConvergenceConfig, or None. Got {type(convergence)}")


# Global API configuration
_API_CONFIG = {
    'api_key': None,
    'api_url': 'https://hyperwave-cloud.onrender.com'
}


def configure_api(api_key: Optional[str] = None, api_url: Optional[str] = None, validate: bool = True) -> Optional[Dict[str, Any]]:
    """Configure API credentials and endpoint, with optional validation.

    Args:
        api_key: API authentication key. If None, uses HYPERWAVE_API_KEY environment variable.
        api_url: API endpoint URL. If None, uses HYPERWAVE_API_URL environment variable
            or defaults to production endpoint.
        validate: If True (default), validates the API key by calling the server.

    Returns:
        Account info dict with 'name', 'email', 'credits_balance' if validation enabled,
        None otherwise.

    Raises:
        ValueError: If no API key is provided and HYPERWAVE_API_KEY is not set.
        RuntimeError: If API key validation fails.

    Example:
        >>> import hyperwave_community as hwc
        >>> account = hwc.configure_api(api_key='your-key-here')
        >>> print(f"Welcome {account['name']}! Credits: {account['credits_balance']}")
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

    # Validate API key if requested
    if validate:
        try:
            response = requests.post(
                f"{_API_CONFIG['api_url']}/account_info",
                params={"api_key": _API_CONFIG['api_key']},
                timeout=10
            )
            if response.status_code == 403:
                raise RuntimeError("Invalid API key. Please check your API key and try again.")
            response.raise_for_status()
            account_info = response.json()
            print(f"✓ API key validated for: {account_info.get('name', 'Unknown')}")
            print(f"  Email: {account_info.get('email', 'N/A')}")
            print(f"  Credits: {account_info.get('credits_balance', 0):.2f}")
            return account_info
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to validate API key: {e}")

    return None


def _get_api_config() -> Dict[str, str]:
    """Get current API configuration."""
    if _API_CONFIG['api_key'] is None:
        if 'HYPERWAVE_API_KEY' in os.environ:
            _API_CONFIG['api_key'] = os.environ['HYPERWAVE_API_KEY']
        else:
            raise RuntimeError(
                "API not configured. Call configure_api() or set HYPERWAVE_API_KEY "
                "environment variable first."
            )
    return _API_CONFIG


def encode_array(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string for API transmission."""
    buffer = io.BytesIO()
    np.save(buffer, arr)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_array(b64_str: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
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
            print(f"Insufficient credits for {operation}.")
            print("Add credits to your account at spinsphotonics.com/billing")
        elif status_code == 429:
            print("Too many concurrent simulations.")
        elif status_code == 502:
            print("Service temporarily unavailable. Please retry.")
        elif status_code == 422:
            print(f"Validation error from API:")
            try:
                error_detail = e.response.json()
                print(f"  {error_detail}")
            except Exception:
                print(f"  {e.response.text}")
        else:
            print(f"Unexpected error (Code: {status_code})")
            try:
                print(f"  Response: {e.response.text[:500]}")
            except Exception:
                pass
    else:
        print("Communication error.")


# =============================================================================
# ACCOUNT INFO
# =============================================================================

def get_account_info(quiet: bool = False) -> Optional[Dict[str, Any]]:
    """Verify API key and get account information including credit balance.

    Args:
        quiet: If True, don't print the greeting message.

    Returns:
        Dictionary with valid, name, email, api_key_prefix, credits_balance, credits_balance_usd.
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
        info = response.json()

        if not quiet:
            name = info.get('name', 'User')
            email = info.get('email', '')
            prefix = info.get('api_key_prefix', '')
            balance = info.get('credits_balance', 0)

            print()
            print(f"  Welcome back, {name}!")
            print(f"  {email}")
            print()
            print(f"  API Key:  {prefix}...")
            print(f"  Credits:  {balance:.2f}")
            print()

        return info

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
    """Estimate simulation cost before running (no auth required)."""
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
        response = requests.post(f"{API_URL}/estimate_cost", json=request_data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error estimating cost: {e}")
        return None


# =============================================================================
# TWO-STAGE WORKFLOW (Recommended)
# =============================================================================

def prepare_simulation(
    device_type: str,
    pdk_config: Dict[str, Any],
    source_port: str = "o1",
    wavelength_um: float = 1.55,
    cells_per_wavelength: int = 25,
    mode_num: int = 0,
    device_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Stage 1: Prepare simulation inputs on Modal CPU.

    This performs all setup work needed before simulation:
    - Build structure recipe from GDSFactory component
    - Create monitors at port locations
    - Solve for waveguide mode at source port
    - Return all data needed for simulation

    Args:
        device_type: Device type name (e.g., "mmi2x2", "mmi1x2").
        pdk_config: PDK configuration dict with material parameters:
            - n_core: Core refractive index (e.g., 3.48 for silicon)
            - n_clad: Cladding refractive index (e.g., 1.45 for SiO2)
            - wg_height_um: Waveguide height in micrometers (e.g., 0.22)
            - clad_top_um: Top cladding thickness in micrometers (e.g., 1.89)
            - clad_bot_um: Bottom cladding (BOX) thickness in micrometers (e.g., 2.0)
        source_port: Input port name (e.g., "o1", "o2").
        wavelength_um: Wavelength in micrometers (default: 1.55).
        cells_per_wavelength: FDTD resolution (default: 25). Higher = more accurate but slower.
        mode_num: Mode number to solve (0 = fundamental, default: 0).
        device_params: Optional dict of device-specific parameters.

    Returns:
        Dict with:
        - mode_preview: Mode visualization data (n_eff, Ex, Ey, Ez)
        - setup_data: Pre-computed setup for run_simulation()
        - wavelength_um: Wavelength used
        - resolution_nm: Resolution in nanometers
        - source_port: Actual source port name

    Example:
        >>> pdk_config = {
        ...     "n_core": 3.48,
        ...     "n_clad": 1.45,
        ...     "wg_height_um": 0.22,
        ...     "clad_top_um": 1.89,
        ...     "clad_bot_um": 2.0,
        ... }
        >>> setup = hwc.prepare_simulation(
        ...     device_type="mmi2x2",
        ...     pdk_config=pdk_config,
        ...     source_port="o1",
        ...     wavelength_um=1.55,
        ...     cells_per_wavelength=25,
        ... )
        >>> print(f"Mode n_eff: {setup['mode_preview']['n_eff']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    # Validate pdk_config
    required_fields = ["n_core", "n_clad", "wg_height_um", "clad_top_um", "clad_bot_um"]
    missing = [f for f in required_fields if f not in pdk_config]
    if missing:
        raise ValueError(f"pdk_config missing required fields: {missing}")

    print(f"Preparing simulation for {device_type}...")
    print(f"  Source: {source_port}, Wavelength: {wavelength_um} um, Resolution: {cells_per_wavelength} cells/wavelength")

    body = {
        "device_type": device_type,
        "device_params": device_params or {},
        "source_port": source_port,
        "wavelength_um": wavelength_um,
        "cells_per_wavelength": cells_per_wavelength,
        "pdk_config": pdk_config,
        "mode_num": mode_num,
    }

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{API_URL}/prepare_simulation",
            json=body,
            headers=headers,
            timeout=600
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") != "success":
            raise Exception(f"Failed to prepare simulation: {result}")

        mode_preview = result.get("mode_preview", {})
        n_eff = mode_preview.get("n_eff", "N/A")
        print(f"Mode solved: n_eff={n_eff}")
        print(f"Setup complete. Ready for run_simulation()")

        return {
            "mode_preview": mode_preview,
            "setup_data": result.get("setup_data"),
            "wavelength_um": result.get("wavelength_um"),
            "resolution_nm": result.get("resolution_nm"),
            "source_port": result.get("source_port"),
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "prepare_simulation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error preparing simulation: {e}")
        return None


def run_simulation(
    device_type: str,
    # Option 1: Pass individual granular results (recommended)
    recipe_result: Optional[Dict[str, Any]] = None,
    monitor_result: Optional[Dict[str, Any]] = None,
    freq_result: Optional[Dict[str, Any]] = None,
    source_result: Optional[Dict[str, Any]] = None,
    # Option 2: Pass pre-packaged setup_data (legacy/advanced)
    setup_data: Optional[Dict[str, Any]] = None,
    # Simulation parameters
    num_steps: int = 20000,
    gpu_type: str = "H100",
    convergence: Optional[str] = "default",
    absorption_widths: List[int] = None,
    absorption_coeff: float = 0.0006173770394704579,
    source_ramp_periods: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Run FDTD simulation on Modal GPU.

    Two ways to call this function:

    **Option 1: Pass granular results directly (recommended)**
    ```python
    results = hwc.run_simulation(
        device_type="mmi2x2",
        recipe_result=recipe_result,
        monitor_result=monitor_result,
        freq_result=freq_result,
        source_result=source_result,
        gpu_type="H100",
    )
    ```

    **Option 2: Pass pre-packaged setup_data (legacy)**
    ```python
    results = hwc.run_simulation(
        device_type="mmi2x2",
        setup_data=setup_data,
        gpu_type="H100",
    )
    ```

    Args:
        device_type: Device type name (for tracking).
        recipe_result: Result from build_recipe().
        monitor_result: Result from build_monitors().
        freq_result: Result from compute_freq_band().
        source_result: Result from solve_mode_source().
        setup_data: Pre-packaged setup (alternative to individual results).
        num_steps: Maximum FDTD steps (default: 20000).
        gpu_type: GPU type - "B200", "H200", "H100", "A100-80GB", etc.
        convergence: Early stopping behavior. Options:
            - "quick": Stop early, check less frequently (fastest)
            - "default": Balanced approach (recommended)
            - "thorough": Check carefully before stopping (most conservative)
            - "full": No early stopping, run all num_steps
            - ConvergenceConfig: Custom configuration object
        absorption_widths: Absorber widths [x, y, z] in cells (default: [82, 40, 40]).
        absorption_coeff: Absorber coefficient.
        source_ramp_periods: Source ramp-up periods (default: 10.0).

    Returns:
        Dict with simulation results:
        - sim_time: GPU simulation time in seconds
        - total_time: Total execution time including overhead
        - monitor_data: Decoded monitor field data
        - powers: Power at each monitor
        - converged: Whether simulation converged (False if convergence="full")
        - convergence_step: Step at which convergence was detected
        - performance: Simulation performance (pts*steps/s)
    """
    import time
    start_time = time.time()

    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    # Build setup_data from individual results if provided
    if recipe_result is not None and monitor_result is not None and source_result is not None:
        # Package granular results into setup_data format
        setup_data = {
            'structure_recipe': recipe_result['recipe'],
            'source_field_b64': encode_array(source_result['source_field']),
            'source_field_shape': list(source_result['source_field'].shape),
            'source_offset': list(source_result['source_offset']),
            'freq_band': list(freq_result['freq_band']) if freq_result else [0.081, 0.081, 1],
            'monitors': monitor_result['monitors'],
            'monitor_names': monitor_result['monitor_names'],
            'dimensions': list(recipe_result['dimensions']),
        }
    elif setup_data is None:
        raise ValueError(
            "Either provide individual results (recipe_result, monitor_result, freq_result, source_result) "
            "or a pre-packaged setup_data dict."
        )
    else:
        # Handle legacy formats: full prepare_simulation result or just setup_data
        if "setup_data" in setup_data and "source_field_base64" not in setup_data:
            setup_data = setup_data["setup_data"]

    # Resolve convergence configuration
    conv_config = _resolve_convergence(convergence)
    use_early_stopping = conv_config is not None

    endpoint = "/early_stopping" if use_early_stopping else "/simulate"
    convergence_name = convergence if isinstance(convergence, str) else ("custom" if conv_config else "full")

    print(f"Starting simulation for {device_type}...")
    print(f"  GPU: {gpu_type}, Max steps: {num_steps}")
    print(f"  Convergence: {convergence_name}")

    # Build request body
    body = {
        "structure_recipe": setup_data.get("structure_recipe"),
        "source_field_b64": setup_data.get("source_field_b64"),
        "source_field_shape": setup_data.get("source_field_shape"),
        "source_offset": setup_data.get("source_offset"),
        "freq_band": setup_data.get("freq_band"),
        "monitors": setup_data.get("monitors"),
        "max_steps": num_steps,
        "check_every_n": conv_config.check_every_n if conv_config else 1000,
        "source_ramp_periods": source_ramp_periods,
        "gpu_type": gpu_type,
        "add_absorption": True,
        "absorption_widths": list(absorption_widths) if absorption_widths else [82, 40, 40],
        "absorption_coeff": float(absorption_coeff),
    }

    # Add early stopping specific parameters
    if use_early_stopping:
        body["relative_threshold"] = conv_config.relative_threshold
        body["absolute_threshold"] = 1e-10  # Fixed value
        body["significant_power_threshold"] = conv_config.power_threshold
        body["min_stable_checks"] = conv_config.min_stable_checks
        # Note: min_steps is handled by the Modal function internally

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        # Call appropriate endpoint
        print(f"Calling {endpoint} API...")
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=body,
            headers=headers,
            timeout=600  # 10 minute timeout for long simulations
        )
        response.raise_for_status()
        result = response.json()

        sim_time = result.get("sim_time", 0)
        total_time = time.time() - start_time
        converged = result.get("converged", False)

        if use_early_stopping:
            convergence_step = result.get("convergence_step", 0)
            print(f"Simulation completed in {sim_time:.1f}s (total: {total_time:.1f}s)")
            if converged:
                print(f"  Converged at step {convergence_step}")
        else:
            print(f"Simulation completed in {sim_time:.1f}s (total: {total_time:.1f}s)")

        # Decode base64-encoded arrays from API response
        def decode_b64_dict(d, shapes=None):
            """Decode base64 strings in a dict back to numpy arrays."""
            import numpy as np
            decoded = {}
            for k, v in d.items():
                if isinstance(v, str) and v:
                    try:
                        # Decode base64 to bytes, then to numpy array
                        arr_bytes = base64.b64decode(v)
                        # Try to infer dtype - most simulation data is float32 or complex64
                        if shapes and k in shapes:
                            shape = shapes[k]
                            # Check if it's complex data (monitor fields are complex)
                            try:
                                arr = np.frombuffer(arr_bytes, dtype=np.complex64).reshape(shape)
                            except ValueError:
                                try:
                                    arr = np.frombuffer(arr_bytes, dtype=np.float32).reshape(shape)
                                except ValueError:
                                    arr = np.frombuffer(arr_bytes, dtype=np.float32)
                        else:
                            # For powers/transmissions, assume float32
                            arr = np.frombuffer(arr_bytes, dtype=np.float32)
                        decoded[k] = arr
                    except Exception:
                        decoded[k] = v  # Keep original if decoding fails
                else:
                    decoded[k] = v
            return decoded

        # Get shapes for decoding monitor data
        monitor_shapes = result.get("monitor_data_shapes", {})

        # Decode the base64-encoded data
        monitor_data = decode_b64_dict(result.get("monitor_data_b64", {}), monitor_shapes)
        powers = decode_b64_dict(result.get("powers", {}))

        # Process results
        return {
            "sim_time": sim_time,
            "total_time": total_time,
            "converged": converged,
            "convergence_step": result.get("convergence_step", 0) if use_early_stopping else 0,
            "monitor_data": monitor_data,
            "monitor_data_shapes": monitor_shapes,
            "monitor_names": result.get("monitor_names", {}),
            "powers": powers,
            "performance": result.get("performance", 0),
        }

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "run_simulation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error running simulation: {e}")
        return None


# =============================================================================
# ONE-SHOT WORKFLOW
# =============================================================================

def simulate(
    device_type: str,
    pdk_config: Dict[str, Any],
    source_port: str = "o1",
    wavelength_um: float = 1.55,
    wavelength_span_nm: float = 100,
    num_wavelengths: int = 5,
    cells_per_wavelength: int = 25,
    num_steps: int = 30000,
    check_every_n: int = 1000,
    source_ramp_periods: float = 5.0,
    gpu_type: str = "H100",
    device_params: Optional[Dict[str, Any]] = None,
    min_steps: int = 0,
    min_stable_checks: int = 3,
    absorber_width: Optional[int] = None,
    absorber_coeff: Optional[float] = None,
    significant_power_threshold: float = 1e-6,
    required_ports: Optional[List[str]] = None,
    poll_interval: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """One-shot FDTD simulation on Modal GPUs (combines setup + simulation).

    NOTE: For better control, use the two-stage workflow instead:
        1. prepare_simulation() - Get mode preview and setup data
        2. run_simulation() - Run FDTD with pre-computed setup

    Args:
        device_type: Device type name (e.g., "mmi2x2", "mmi1x2").
        pdk_config: PDK configuration dict with material parameters:
            - n_core: Core refractive index (e.g., 3.48 for silicon)
            - n_clad: Cladding refractive index (e.g., 1.45 for SiO2)
            - wg_height_um: Waveguide height in micrometers (e.g., 0.22)
            - clad_top_um: Top cladding thickness in micrometers (e.g., 1.89)
            - clad_bot_um: Bottom cladding (BOX) thickness in micrometers (e.g., 2.0)
        source_port: Input port name (e.g., "o1", "o2").
        wavelength_um: Center wavelength in micrometers (default: 1.55).
        wavelength_span_nm: Wavelength span in nanometers (default: 100).
        num_wavelengths: Number of wavelength points (default: 5).
        cells_per_wavelength: FDTD resolution (default: 25).
        num_steps: Maximum FDTD steps (default: 30000).
        check_every_n: Convergence check interval (default: 1000).
        source_ramp_periods: Source ramp-up periods (default: 5.0).
        gpu_type: GPU type - "B200", "H200", "H100", "A100-80GB", etc.
        device_params: Optional dict of device-specific parameters.
        min_steps: Minimum FDTD steps before convergence check.
        min_stable_checks: Required consecutive stable checks.
        absorber_width: PML absorber width in pixels.
        absorber_coeff: PML absorption coefficient.
        significant_power_threshold: Min power for port detection.
        required_ports: List of port names that must have power.
        poll_interval: Seconds between status polls.

    Returns:
        Dict with simulation results:
        - s_parameters: Analyzed transmission data
        - field_intensity: 2D field for visualization
        - sim_time: GPU simulation time in seconds
        - converged: Whether simulation converged

    Example:
        >>> pdk_config = {
        ...     "n_core": 3.48,
        ...     "n_clad": 1.45,
        ...     "wg_height_um": 0.22,
        ...     "clad_top_um": 1.89,
        ...     "clad_bot_um": 2.0,
        ... }
        >>> results = hwc.simulate(
        ...     device_type="mmi2x2",
        ...     pdk_config=pdk_config,
        ...     source_port="o1",
        ...     wavelength_um=1.55,
        ... )
        >>> print(f"T_total: {results['s_parameters']['T_total']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    # Validate pdk_config
    required_fields = ["n_core", "n_clad", "wg_height_um", "clad_top_um", "clad_bot_um"]
    missing = [f for f in required_fields if f not in pdk_config]
    if missing:
        raise ValueError(f"pdk_config missing required fields: {missing}")

    print(f"Starting simulation for {device_type}...")
    print(f"  Source: {source_port}, Wavelength: {wavelength_um} um (+/- {wavelength_span_nm/2} nm)")
    print(f"  GPU: {gpu_type}")

    body = {
        "device_type": device_type,
        "source_port": source_port,
        "wavelength_um": wavelength_um,
        "wavelength_span_nm": wavelength_span_nm,
        "num_wavelengths": num_wavelengths,
        "cells_per_wavelength": cells_per_wavelength,
        "num_steps": num_steps,
        "check_every_n": check_every_n,
        "source_ramp_periods": source_ramp_periods,
        "gpu_type": gpu_type,
        "pdk_config": pdk_config,
        "device_params": device_params or {},
        "min_steps": min_steps,
        "min_stable_checks": min_stable_checks,
        "absorber_width": absorber_width,
        "absorber_coeff": absorber_coeff,
        "significant_power_threshold": significant_power_threshold,
        "required_ports": required_ports,
    }

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        # Start the job
        response = requests.post(
            f"{API_URL}/simulate/start",
            json=body,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") != "success":
            raise Exception(f"Failed to start simulation: {result}")

        job_id = result.get("job_id")
        print(f"Job started: {job_id[:8]}...")

        # Poll for completion
        last_progress = ""
        while True:
            time.sleep(poll_interval)

            status_response = requests.get(
                f"{API_URL}/simulate/status/{job_id}",
                headers=headers,
                timeout=30
            )
            status_result = status_response.json()

            status = status_result.get("status")
            progress = status_result.get("progress", "")

            if progress and progress != last_progress:
                print(f"  {progress}")
                last_progress = progress

            if status == "completed":
                sim_result = status_result.get("result", {})
                sim_time = sim_result.get("sim_time", 0)
                converged = sim_result.get("converged", False)
                print(f"Simulation completed in {sim_time:.1f}s (converged: {converged})")
                return sim_result

            elif status in ("failed", "error"):
                error = status_result.get("error", "Unknown error")
                print(f"Simulation failed: {error}")
                raise Exception(f"Simulation failed: {error}")

            elif status not in ("starting", "running", "pending"):
                raise Exception(f"Unexpected status: {status}")

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "simulate")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error running simulation: {e}")
        return None


# =============================================================================
# GRANULAR WORKFLOW FUNCTIONS
# =============================================================================

def build_recipe(
    component_name: str,
    component_kwargs: Optional[Dict[str, Any]] = None,
    extension_length: float = 2.0,
    resolution_nm: float = 30.0,
    n_core: float = 3.48,
    n_clad: float = 1.45,
    wg_height_um: float = 0.22,
    total_height_um: float = 4.0,
    padding: Tuple[int, int, int, int] = (100, 100, 0, 0),
    density_radius: int = 3,
    vertical_radius: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """Build structure recipe from GDSFactory component on Modal CPU.

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

    Returns:
        Dictionary containing recipe, density_core, density_clad, dimensions,
        port_info, layer_config, eps_values, resolution_um.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Building recipe for {component_name}...")

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

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/build_recipe",
            json=request_data,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        results = response.json()

        dims = results.get('dimensions', [])
        ports = list(results.get('port_info', {}).keys())
        print(f"Recipe built: {dims[0]}x{dims[1]}x{dims[2]} cells")
        print(f"Ports: {ports}")

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
    source_port: str = "o1",
    monitor_x_um: float = 0.1,
    monitor_y_um: float = 1.5,
    monitor_z_um: float = 1.5,
    resolution_um: float = 0.03,
    source_offset_cells: int = 5,
    structure_recipe: Dict[str, Any] = None,
    show_structure: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build monitors from port information on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        port_info: Port information from build_recipe()
        dimensions: Structure dimensions (Lx, Ly, Lz)
        source_port: Name of port to use as source
        monitor_x_um: Monitor thickness in x (propagation direction)
        monitor_y_um: Monitor size in y
        monitor_z_um: Monitor size in z
        resolution_um: Grid resolution in micrometers
        source_offset_cells: Offset of source from monitor
        structure_recipe: Recipe from build_recipe() - required for visualization
        show_structure: If True and structure_recipe provided, show structure plot
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Building monitors (source: {source_port})...")

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

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/build_monitors",
            json=request_data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        print(f"Monitors built: {list(result.get('monitor_names', {}).keys())}")

        # Visualize structure if requested and recipe provided
        if show_structure and structure_recipe is not None:
            visualize_structure(
                structure_recipe=structure_recipe,
                monitors=result['monitors'],
                monitor_names=result['monitor_names'],
                dimensions=dimensions,
                source_position=result['source_position'],
                axis='z',
                show=True,
                figsize=(14, 3),
            )

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "monitor building")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error building monitors: {e}")
        return None


def compute_freq_band(
    wl_min_um: float = 1.55,
    wl_max_um: float = 1.55,
    n_freqs: int = 1,
    resolution_um: float = 0.03,
) -> Optional[Dict[str, Any]]:
    """Compute frequency band from wavelength range on Modal CPU.

    This function runs on CPU and does NOT consume credits.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Computing freq band: {wl_min_um}-{wl_max_um} um ({n_freqs} points)...")

    request_data = {
        "wl_min_um": wl_min_um,
        "wl_max_um": wl_max_um,
        "n_freqs": n_freqs,
        "resolution_um": resolution_um,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/compute_freq_band",
            json=request_data,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        results = response.json()
        print(f"Freq band computed: {results['freq_band']}")

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
    show_mode: bool = True,
) -> Optional[Dict[str, Any]]:
    """Solve for waveguide mode source field on Modal CPU.

    This function runs on CPU and does NOT consume credits.

    Args:
        density_core: Core density array from build_recipe()
        density_clad: Cladding density array from build_recipe()
        source_x_position: X position for source plane
        mode_bounds: Mode bounds from build_monitors()
        layer_config: Layer configuration from build_recipe()
        eps_values: Permittivity values (eps_core, eps_clad)
        freq_band: Frequency band (f_min, f_max, n_freqs)
        slice_half_width: Half-width of mode slice in cells
        mode_num: Mode number (0 = fundamental)
        propagation_axis: Propagation axis ('x', 'y', or 'z')
        show_mode: If True, display mode profile visualization
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Solving mode at x={source_x_position}...")

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

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/solve_mode_source",
            json=request_data,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        results = response.json()

        mode_info = results.get('mode_info', {})
        n_eff = mode_info.get('n_eff', 'N/A')
        print(f"Mode solved: n_eff={n_eff}")

        source_field = decode_array(results['source_field_b64'])
        source_offset = tuple(results['source_offset'])

        result = {
            'source_field': source_field,
            'source_offset': source_offset,
            'mode_info': mode_info,
            'freq_band': tuple(results['freq_band']),
            'solve_time_seconds': results.get('solve_time_seconds', 0.0),
        }

        # Visualize mode if requested
        if show_mode:
            visualize_mode_source(
                source_field=source_field,
                source_offset=source_offset,
                show=True,
            )

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "mode source solving")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error solving mode source: {e}")
        return None


def get_default_absorber_params(
    structure_dimensions: Tuple[int, int, int],
    absorber_fraction: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Get default absorber parameters based on structure size on Modal CPU.

    This function runs on CPU and does NOT consume credits.
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Computing absorber params...")

    request_data = {
        "structure_dimensions": list(structure_dimensions),
        "absorber_fraction": absorber_fraction,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/get_default_absorber_params",
            json=request_data,
            headers=headers,
            timeout=120  # Modal cold start can take time
        )
        response.raise_for_status()
        results = response.json()

        print(f"Absorber params computed")

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


# =============================================================================
# NEW GRANULAR WORKFLOW FUNCTIONS (Fine-grained control)
# =============================================================================

def load_component(
    name: str,
    extension_length: float = 2.0,
    component_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Load GDSFactory component and get its metadata.

    This function runs on CPU and does NOT consume credits.

    Args:
        name: Component name from gdsfactory (e.g., "mmi1x2", "mmi2x2").
        extension_length: Length to extend ports in micrometers (default: 2.0).
        component_kwargs: Optional kwargs to pass to component constructor.

    Returns:
        Dictionary containing:
        - name: Component name
        - port_info: Port information dict
        - bounding_box_um: Bounding box in micrometers
        - component_params: Component parameters

    Example:
        >>> component_data = hwc.load_component("mmi2x2", extension_length=2.0)
        >>> print(f"Ports: {list(component_data['port_info'].keys())}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Loading component: {name}...")

    request_data = {
        "name": name,
        "extension_length": extension_length,
        "component_kwargs": component_kwargs or {},
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/granular/component/load",
            json=request_data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        ports = list(result.get('port_info', {}).keys())
        print(f"Component loaded: {ports}")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "component loading")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error loading component: {e}")
        return None


def create_structure_recipe(
    component_data: Dict[str, Any],
    resolution_nm: float = 20.0,
    n_core: float = 3.48,
    n_clad: float = 1.4457,
    wg_height_um: float = 0.22,
    clad_top_um: float = 1.89,
    clad_bot_um: float = 2.0,
    padding: Tuple[int, int, int, int] = (100, 100, 0, 0),
    density_radius: int = 3,
    vertical_radius: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """Create structure recipe from component data.

    This function runs on CPU and does NOT consume credits.

    Args:
        component_data: Component data from load_component().
        resolution_nm: Grid resolution in nanometers (default: 20.0).
        n_core: Core refractive index (default: 3.48 for silicon).
        n_clad: Cladding refractive index (default: 1.4457 for SiO2).
        wg_height_um: Waveguide height in micrometers (default: 0.22).
        clad_top_um: Top cladding thickness in micrometers (default: 1.89).
        clad_bot_um: Bottom cladding thickness in micrometers (default: 2.0).
        padding: (left, right, top, bottom) padding in theta pixels.
        density_radius: Radius for density filtering (default: 3).
        vertical_radius: Vertical blur radius (default: 2.0).

    Returns:
        Dictionary containing:
        - structure_recipe: Recipe for simulation
        - dimensions: (nx, ny, nz) grid dimensions
        - port_info_cells: Port locations in grid cells
        - freq_band: Frequency band tuple
        - wavelengths_um: Wavelength list

    Example:
        >>> component_data = hwc.load_component("mmi2x2")
        >>> recipe_data = hwc.create_structure_recipe(
        ...     component_data,
        ...     resolution_nm=20,
        ...     n_core=3.48,
        ...     n_clad=1.4457,
        ... )
        >>> print(f"Dimensions: {recipe_data['dimensions']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Creating structure recipe...")

    request_data = {
        "component_data": component_data,
        "resolution_nm": resolution_nm,
        "n_core": n_core,
        "n_clad": n_clad,
        "wg_height_um": wg_height_um,
        "clad_top_um": clad_top_um,
        "clad_bot_um": clad_bot_um,
        "padding": list(padding),
        "density_radius": density_radius,
        "vertical_radius": vertical_radius,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/granular/structure/create",
            json=request_data,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()

        dims = result.get('dimensions', [])
        print(f"Structure recipe created: {dims[0]}x{dims[1]}x{dims[2]} cells")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "structure recipe creation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error creating structure recipe: {e}")
        return None


def create_monitors(
    structure_recipe_data: Dict[str, Any],
    source_port: str = "o2",
    monitor_x_um: float = 0.1,
    monitor_y_um: float = 1.5,
    monitor_z_um: float = 1.5,
    source_offset_cells: int = 5,
) -> Optional[Dict[str, Any]]:
    """Create monitors from structure recipe data.

    This function runs on CPU and does NOT consume credits.

    Args:
        structure_recipe_data: Structure recipe data from create_structure_recipe().
        source_port: Name of input port (default: "o2").
        monitor_x_um: Monitor width in x direction in micrometers (default: 0.1).
        monitor_y_um: Monitor width in y direction in micrometers (default: 1.5).
        monitor_z_um: Monitor width in z direction in micrometers (default: 1.5).
        source_offset_cells: Source offset from port in cells (default: 5).

    Returns:
        Dictionary containing:
        - monitors: Monitor configuration dict
        - mode_solve_params: Parameters for mode solving
        - source_port_name: Actual source port name

    Example:
        >>> monitors_data = hwc.create_monitors(
        ...     structure_recipe_data,
        ...     source_port="o2",
        ...     monitor_x_um=0.1,
        ... )
        >>> print(f"Source port: {monitors_data['source_port_name']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Creating monitors (source: {source_port})...")

    request_data = {
        "structure_recipe_data": structure_recipe_data,
        "source_port": source_port,
        "monitor_x_um": monitor_x_um,
        "monitor_y_um": monitor_y_um,
        "monitor_z_um": monitor_z_um,
        "source_offset_cells": source_offset_cells,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/granular/monitors/create",
            json=request_data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        print(f"Monitors created")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "monitor creation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error creating monitors: {e}")
        return None


def solve_mode(
    structure_recipe: Dict[str, Any],
    mode_solve_params: Dict[str, Any],
    freq_band: Tuple[float, float, int],
    mode_num: int = 0,
) -> Optional[Dict[str, Any]]:
    """Solve for waveguide mode at source port.

    This function runs on CPU and does NOT consume credits.

    Args:
        structure_recipe: Structure recipe dict.
        mode_solve_params: Mode solve parameters from create_monitors().
        freq_band: Frequency band tuple (f_min, f_max, n_freqs).
        mode_num: Mode number to solve (0 = fundamental, default: 0).

    Returns:
        Dictionary containing:
        - source_field: Source field numpy array
        - source_offset: Source offset tuple
        - freq_band: Frequency band used
        - mode_info: Mode information dict (n_eff, etc.)

    Example:
        >>> source_data = hwc.solve_mode(
        ...     structure_recipe,
        ...     mode_solve_params,
        ...     freq_band=(1.2, 1.3, 5),
        ...     mode_num=0,
        ... )
        >>> print(f"n_eff: {source_data['mode_info']['n_eff']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Solving mode (mode_num={mode_num})...")

    request_data = {
        "structure_recipe": structure_recipe,
        "mode_solve_params": mode_solve_params,
        "freq_band": list(freq_band),
        "mode_num": mode_num,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/granular/mode/solve",
            json=request_data,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()

        # Decode base64 source field
        if 'source_field_b64' in result:
            result['source_field'] = decode_array(result['source_field_b64'])
            del result['source_field_b64']

        mode_info = result.get('mode_info', {})
        n_eff = mode_info.get('n_eff', 'N/A')
        print(f"Mode solved: n_eff={n_eff}")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "mode solving")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error solving mode: {e}")
        return None


def run_gpu_simulation(
    structure_recipe: Dict[str, Any],
    source_data: Dict[str, Any],
    monitors: Dict[str, Any],
    gpu_type: str = "H100",
    max_steps: int = 20000,
    check_every_n: int = 1000,
    source_ramp_periods: float = 10.0,
    min_steps: int = 0,
    min_stable_checks: int = 3,
    absorber_width: Optional[int] = None,
    absorber_coeff: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Run FDTD simulation on GPU with granular inputs.

    This function runs on GPU and DOES consume credits.

    Args:
        structure_recipe: Structure recipe dict.
        source_data: Source data from solve_mode().
        monitors: Monitor configuration from create_monitors().
        gpu_type: GPU type - "H100", "A100-80GB", etc. (default: "H100").
        max_steps: Maximum FDTD steps (default: 20000).
        check_every_n: Convergence check interval (default: 1000).
        source_ramp_periods: Source ramp-up periods (default: 10.0).
        min_steps: Minimum steps before early stopping (default: 0).
        min_stable_checks: Consecutive stable checks for convergence (default: 3).
        absorber_width: Absorber width in cells (optional).
        absorber_coeff: Absorber coefficient (optional).

    Returns:
        Dictionary containing:
        - field_data: Field data with decoded numpy arrays
        - s_parameters: Transmission/reflection data
        - sim_time: GPU simulation time
        - converged: Whether simulation converged

    Example:
        >>> results = hwc.run_gpu_simulation(
        ...     structure_recipe,
        ...     source_data,
        ...     monitors,
        ...     gpu_type="H100",
        ...     max_steps=20000,
        ... )
        >>> print(f"Converged: {results['converged']}")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Starting GPU simulation...")
    print(f"  GPU: {gpu_type}, Max steps: {max_steps}")

    # Encode source field if it's a numpy array
    request_data = {
        "structure_recipe": structure_recipe,
        "source_data": source_data.copy(),
        "monitors": monitors,
        "gpu_type": gpu_type,
        "max_steps": max_steps,
        "check_every_n": check_every_n,
        "source_ramp_periods": source_ramp_periods,
        "min_steps": min_steps,
        "min_stable_checks": min_stable_checks,
    }

    if absorber_width is not None:
        request_data["absorber_width"] = absorber_width
    if absorber_coeff is not None:
        request_data["absorber_coeff"] = absorber_coeff

    # Encode source field if present
    if 'source_field' in request_data['source_data']:
        source_field = request_data['source_data']['source_field']
        if isinstance(source_field, np.ndarray):
            request_data['source_data']['source_field_b64'] = encode_array(source_field)
            del request_data['source_data']['source_field']

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/granular/simulation/run",
            json=request_data,
            headers=headers,
            timeout=1800  # 30 minutes for long simulations
        )
        response.raise_for_status()
        result = response.json()

        # Decode base64 arrays in response
        if 'field_data' in result:
            field_data = result['field_data']
            for key in field_data:
                if isinstance(field_data[key], str) and key.endswith('_b64'):
                    array_key = key.replace('_b64', '')
                    field_data[array_key] = decode_array(field_data[key])
                    del field_data[key]

        sim_time = result.get('sim_time', 0)
        converged = result.get('converged', False)
        print(f"Simulation completed in {sim_time:.1f}s (converged: {converged})")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "GPU simulation")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error running GPU simulation: {e}")
        return None


def analyze_transmission(
    simulation_results: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Analyze transmission from simulation results.

    This function runs on CPU and does NOT consume credits.

    Args:
        simulation_results: Simulation results from run_gpu_simulation().

    Returns:
        Dictionary containing:
        - wavelengths_nm: Wavelengths in nanometers
        - transmission: Transmission values per port
        - total_transmission: Total transmission sum
        - excess_loss_dB: Excess loss in dB

    Example:
        >>> trans_data = hwc.analyze_transmission(simulation_results)
        >>> print(f"Excess loss: {trans_data['excess_loss_dB']:.2f} dB")
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Analyzing transmission...")

    request_data = {
        "simulation_results": simulation_results,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/granular/analysis/transmission",
            json=request_data,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        print(f"Transmission analyzed")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "transmission analysis")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error analyzing transmission: {e}")
        return None


def get_field_slice(
    simulation_results: Dict[str, Any],
    monitor_name: str = "xy_mid",
    freq_idx: int = 0,
) -> Optional[Dict[str, Any]]:
    """Get 2D field slice from simulation results.

    This function runs on CPU and does NOT consume credits.

    Args:
        simulation_results: Simulation results from run_gpu_simulation().
        monitor_name: Name of monitor to extract (default: "xy_mid").
        freq_idx: Frequency index to extract (default: 0).

    Returns:
        Dictionary containing:
        - intensity_2d: 2D field intensity numpy array
        - shape: Array shape tuple
        - monitor_name: Monitor name used

    Example:
        >>> field_data = hwc.get_field_slice(
        ...     simulation_results,
        ...     monitor_name="xy_mid",
        ...     freq_idx=0,
        ... )
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(field_data['intensity_2d'])
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Getting field slice ({monitor_name}, freq={freq_idx})...")

    request_data = {
        "simulation_results": simulation_results,
        "monitor_name": monitor_name,
        "freq_idx": freq_idx,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/granular/analysis/field_slice",
            json=request_data,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        # Decode base64 intensity array
        if 'intensity_2d_b64' in result:
            result['intensity_2d'] = decode_array(result['intensity_2d_b64'])
            del result['intensity_2d_b64']

        print(f"Field slice extracted: {result.get('shape')}")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "field slice extraction")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting field slice: {e}")
        return None


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_structure(
    structure_recipe: Dict[str, Any],
    monitors: list,
    monitor_names: Dict[str, int] = None,
    dimensions: tuple = None,
    source_position: int = None,
    axis: str = "y",
    position: int = None,
    show: bool = True,
    save_path: str = None,
    figsize: tuple = (14, 8),
) -> Dict[str, Any]:
    """Generate structure + monitors visualization.

    This function calls Modal to build the 3D structure and generate
    a 2D cross-section visualization with monitor positions overlaid.
    Useful for debugging simulation setup without running locally.

    Args:
        structure_recipe: Recipe dict from build_recipe()
        monitors: List of monitor dicts (from build_monitors)
        monitor_names: Dict mapping names to indices
        dimensions: (Lx, Ly, Lz) structure dimensions
        source_position: X position of source plane (optional)
        axis: Viewing axis ('x', 'y', or 'z'), default 'y'
        position: Slice position along axis (None = middle)
        show: If True, display the image using matplotlib
        save_path: If provided, save PNG to this path
        figsize: Figure size as (width, height)

    Returns:
        Dictionary containing:
            - image_b64: Base64-encoded PNG image
            - dimensions: Structure dimensions
            - slice_info: Dict with axis and position of slice

    Example:
        >>> recipe_result = hwc.build_recipe(component_name="mmi2x2")
        >>> monitor_result = hwc.build_monitors(...)
        >>> result = hwc.visualize_structure(
        ...     structure_recipe=recipe_result['recipe'],
        ...     monitors=monitor_result['monitors'],
        ...     source_position=monitor_result['source_position'],
        ...     axis='y'
        ... )
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    print(f"Generating structure visualization (axis={axis})...")

    request_data = {
        "structure_recipe": structure_recipe,
        "monitors": monitors,
        "monitor_names": monitor_names,
        "dimensions": list(dimensions) if dimensions else None,
        "source_position": source_position,
        "axis": axis,
        "position": position,
        "figsize": list(figsize) if figsize else None,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/visualize/structure",
            json=request_data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        print(f"Visualization generated: {result['dimensions']}")

        # Display image if requested
        if show:
            try:
                import matplotlib.pyplot as plt
                from PIL import Image
                import io

                image_bytes = base64.b64decode(result['image_b64'])
                image = Image.open(io.BytesIO(image_bytes))

                plt.figure(figsize=(14, 8))
                plt.imshow(image)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("Install matplotlib and Pillow to display images: pip install matplotlib Pillow")

        # Save image if requested
        if save_path:
            image_bytes = base64.b64decode(result['image_b64'])
            with open(save_path, 'wb') as f:
                f.write(image_bytes)
            print(f"Image saved to: {save_path}")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "structure visualization")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error generating visualization: {e}")
        return None


def visualize_mode_source(
    source_field: np.ndarray = None,
    source_field_b64: str = None,
    source_field_shape: list = None,
    source_offset: tuple = None,
    show: bool = True,
    save_path: str = None,
) -> Dict[str, Any]:
    """Generate mode source visualization.

    This function calls Modal to visualize the waveguide mode profile,
    showing field intensity and individual field components (Ex, Ey, Ez).

    Args:
        source_field: Source field array (n_freqs, 6, 1, ny, nz). Either this or source_field_b64 required.
        source_field_b64: Base64-encoded source field. Either this or source_field required.
        source_field_shape: Shape of source field (required if using source_field_b64)
        source_offset: Source offset (x, y, z) for display
        show: If True, display the image using matplotlib
        save_path: If provided, save PNG to this path

    Returns:
        Dictionary containing:
            - image_b64: Base64-encoded PNG image
            - field_shape: Shape of the source field
            - max_intensity: Maximum field intensity

    Example:
        >>> source_result = hwc.solve_mode_source(...)
        >>> result = hwc.visualize_mode_source(
        ...     source_field=source_result['source_field'],
        ...     source_offset=source_result['source_offset'],
        ...     show=True,
        ... )
    """
    config = _get_api_config()
    API_URL = config['api_url']
    API_KEY = config['api_key']

    # Encode source field if provided as array
    if source_field is not None:
        source_field_b64 = encode_array(source_field)
        source_field_shape = list(source_field.shape)
    elif source_field_b64 is None:
        raise ValueError("Either source_field or source_field_b64 must be provided")

    print(f"Generating mode source visualization...")

    request_data = {
        "source_field_b64": source_field_b64,
        "source_field_shape": source_field_shape,
        "source_offset": list(source_offset) if source_offset else None,
    }

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{API_URL}/visualize/mode_source",
            json=request_data,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        print(f"Mode visualization generated: shape={result['field_shape']}, max_intensity={result['max_intensity']:.2e}")

        # Display image if requested
        if show:
            try:
                import matplotlib.pyplot as plt
                from PIL import Image
                import io

                image_bytes = base64.b64decode(result['image_b64'])
                image = Image.open(io.BytesIO(image_bytes))

                plt.figure(figsize=(14, 6))
                plt.imshow(image)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("Install matplotlib and Pillow to display images: pip install matplotlib Pillow")

        # Save image if requested
        if save_path:
            image_bytes = base64.b64decode(result['image_b64'])
            with open(save_path, 'wb') as f:
                f.write(image_bytes)
            print(f"Image saved to: {save_path}")

        return result

    except requests.exceptions.HTTPError as e:
        _handle_api_error(e, "mode source visualization")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error generating visualization: {e}")
        return None


# =============================================================================
# Analysis Functions (local, no API calls)
# =============================================================================

def compute_poynting_vector(fields: np.ndarray) -> np.ndarray:
    """Compute Poynting vector from electromagnetic field components.

    Calculates S = 0.5 * Re(E × H*), the time-averaged power flow.

    Args:
        fields: Field array of shape (6, ny, nz) or (6, nx, ny, nz) where
                the first axis contains [Ex, Ey, Ez, Hx, Hy, Hz]

    Returns:
        Poynting vector array of shape (3, ny, nz) or (3, nx, ny, nz)
        containing [Sx, Sy, Sz] components
    """
    Ex, Ey, Ez = fields[0], fields[1], fields[2]
    Hx, Hy, Hz = fields[3], fields[4], fields[5]

    # S = 0.5 * Re(E × H*)
    Sx = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
    Sy = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
    Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))

    return np.stack([Sx, Sy, Sz], axis=0)


def compute_monitor_power(
    monitor_data: np.ndarray,
    direction: str = 'x'
) -> float:
    """Compute total power through a monitor from field data.

    Args:
        monitor_data: Field data from a monitor, shape (n_freqs, 6, ny, nz)
                     or (6, n_samples, ny, nz) depending on monitor type
        direction: Direction of power flow ('x', 'y', or 'z')

    Returns:
        Total power (absolute value) through the monitor
    """
    # Handle different data shapes
    if len(monitor_data.shape) == 4:
        # Shape: (6, n_samples, ny, nz) - average over samples
        fields = monitor_data
        fields_mean = np.mean(fields, axis=1)
    else:
        fields_mean = monitor_data

    S = compute_poynting_vector(fields_mean)

    # Select component based on direction
    direction_map = {'x': 0, 'y': 1, 'z': 2}
    component = direction_map.get(direction.lower(), 0)

    return np.abs(np.sum(S[component, :, :]))


def analyze_transmission(
    results: Dict[str, Any],
    input_monitor: str = "Input_o1",
    output_monitors: List[str] = None,
    direction: str = 'x',
    print_results: bool = True,
) -> Dict[str, Any]:
    """Analyze transmission from simulation results.

    Computes transmission coefficients for each output monitor relative
    to the input monitor using Poynting vector power flow.

    Args:
        results: Simulation results from run_simulation()
        input_monitor: Name of input monitor
        output_monitors: List of output monitor names. If None, auto-detects
                        monitors starting with "Output_"
        direction: Direction of power flow for Poynting vector ('x', 'y', 'z')
        print_results: If True, print formatted results table

    Returns:
        Dictionary with:
            - 'power_in': Input power
            - 'transmissions': Dict mapping monitor name to transmission value
            - 'total_transmission': Sum of all output transmissions
            - 'excess_loss_dB': Excess loss in dB (10*log10(total))
    """
    monitor_data = results.get('monitor_data', {})

    # Auto-detect output monitors if not specified
    if output_monitors is None:
        output_monitors = [name for name in monitor_data.keys()
                         if name.startswith("Output_")]

    # Helper to extract fields from monitor data (handles both formats)
    def _get_fields(data, freq_idx=0):
        """Extract field data, handling both (n_freqs, 6, ny, nz) and (6, ny, nz) formats."""
        arr = np.array(data)
        if arr.ndim == 4:
            # Shape: (n_freqs, 6, ny, nz) - index into frequency axis
            return arr[freq_idx]
        elif arr.ndim == 3:
            # Shape: (6, ny, nz) - use directly
            return arr
        else:
            raise ValueError(f"Unexpected monitor data shape: {arr.shape}. Expected 3D or 4D array.")

    # Compute input power
    if input_monitor not in monitor_data:
        raise ValueError(f"Input monitor '{input_monitor}' not found in results")

    input_fields = _get_fields(monitor_data[input_monitor])
    power_in = compute_monitor_power(input_fields, direction)

    # Compute transmission for each output
    transmissions = {}
    for monitor_name in output_monitors:
        if monitor_name not in monitor_data:
            print(f"Warning: Monitor '{monitor_name}' not found, skipping")
            continue
        output_fields = _get_fields(monitor_data[monitor_name])
        power_out = compute_monitor_power(output_fields, direction)
        transmissions[monitor_name] = power_out / power_in

    total_T = sum(transmissions.values())
    excess_loss_dB = 10 * np.log10(total_T) if total_T > 0 else float('-inf')

    # Print results if requested
    if print_results:
        print(f"{'='*60}")
        print(f"Transmission Analysis (Input: {input_monitor})")
        print(f"{'Monitor':<20} {'Transmission':>12} {'dB':>10}")
        print("-" * 60)
        for name, T in transmissions.items():
            T_dB = 10 * np.log10(T) if T > 0 else float('-inf')
            print(f"{name:<20} {float(T):>12.4f} {float(T_dB):>10.2f}")
        print("-" * 60)
        print(f"{'Total':<20} {float(total_T):>12.4f} {float(excess_loss_dB):>10.2f}")
        print(f"{'='*60}")

    return {
        'power_in': power_in,
        'transmissions': transmissions,
        'total_transmission': total_T,
        'excess_loss_dB': excess_loss_dB,
    }


def get_field_intensity_2d(
    results: Dict[str, Any],
    monitor_name: str = 'xy_mid',
    dimensions: Tuple[int, int, int] = None,
    resolution_um: float = None,
    freq_band: Tuple[float, float, float] = None,
) -> Dict[str, Any]:
    """Extract 2D field intensity from simulation results for plotting.

    Computes |E|² from the specified monitor and provides extent/wavelength
    info for matplotlib plotting.

    Args:
        results: Simulation results from run_simulation()
        monitor_name: Name of the 2D monitor (default: 'xy_mid')
        dimensions: Structure dimensions (Lx, Ly, Lz) for computing extent
        resolution_um: Grid resolution in micrometers
        freq_band: Frequency band tuple (freq_min, freq_max, n_freqs) for wavelength

    Returns:
        Dictionary with:
            - 'intensity': 2D numpy array of |E|² ready for imshow
            - 'extent': [x_min, x_max, y_min, y_max] in μm for imshow extent
            - 'wavelength_nm': Wavelength in nm (if freq_band provided)
    """
    monitor_data = results.get('monitor_data', {})

    if monitor_name not in monitor_data:
        raise ValueError(f"Monitor '{monitor_name}' not found in results")

    data = np.array(monitor_data[monitor_name])

    # Handle different data shapes from various endpoints
    if data.ndim == 3:
        # Shape: (6, ny, nz) - direct field data
        E_fields = data[0:3, :, :]
        field_intensity = np.sum(np.abs(E_fields)**2, axis=0)
        field_2d = field_intensity.T
    elif data.ndim == 4:
        # Shape: (n_freqs, 6, ny, nz) - frequency-indexed data
        # Use first frequency
        E_fields = data[0, 0:3, :, :]
        field_intensity = np.sum(np.abs(E_fields)**2, axis=0)
        field_2d = field_intensity.T
    else:
        raise ValueError(f"Unexpected monitor data shape: {data.shape}. Expected 3D or 4D array.")

    result = {'intensity': field_2d}

    # Compute extent if dimensions and resolution provided
    if dimensions is not None and resolution_um is not None:
        Lx, Ly, Lz = dimensions
        x_um = Lx * resolution_um
        y_um = Ly * resolution_um
        result['extent'] = [-x_um/2, x_um/2, -y_um/2, y_um/2]

    # Compute wavelength if freq_band provided
    if freq_band is not None and resolution_um is not None:
        freq_at_idx = freq_band[0]
        wl_nm = 2 * np.pi / freq_at_idx * resolution_um * 1000
        result['wavelength_nm'] = wl_nm

    return result
