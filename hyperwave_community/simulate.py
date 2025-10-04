"""Simulation utilities and visualization functions.

This module provides helper functions for running simulations and visualizing results.

Main functions:
    simulate: Run FDTD simulation on GPU via API
    quick_view_monitors: Quick visualization of monitor field data
"""

import numpy as np
import jax.numpy as jnp
import requests
from typing import Dict, Any, Tuple, Optional

from .api_client import encode_array, decode_array


def simulate(
    structure,
    source_field: jnp.ndarray,
    source_offset: Tuple[int, int, int],
    freq_band: Tuple[float, float, int],
    monitors,
    mode_info: Optional[Dict] = None,
    simulation_steps: int = 10000,
    check_every_n: int = 1000,
    source_ramp_periods: float = 5.0,
    add_absorption: bool = True,
    absorption_widths: Tuple[int, int, int] = (70, 35, 17),
    absorption_coeff: float = 4.89e-3,
    gpu_type: str = "H100",
    api_key: Optional[str] = None,
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
        simulation_steps: Number of FDTD time steps to run. The simulation will
            converge to a relatively low error at around this step count.
        check_every_n: Convergence check interval (in time steps).
        source_ramp_periods: Number of periods for source turn-on.
        add_absorption: If True, add PML absorption boundaries on GPU.
        absorption_widths: PML widths as (x_width, y_width, z_width) in pixels.
        absorption_coeff: PML absorption coefficient.
        gpu_type: GPU type to use (H100, A100, A10G, L4).
        api_key: API key (overrides configured key).

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
    # Check for API key
    if not api_key:
        print("API key required to proceed.")
        print("Sign up for free at spinsphotonics.com to get your API key.")
        return None

    API_URL = "https://hyperwave-cloud.onrender.com"

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
        "simulation_steps": simulation_steps,
        "check_every_n": check_every_n,
        "source_ramp_periods": source_ramp_periods,
        "add_absorption": add_absorption,
        "absorption_widths": list(absorption_widths),
        "absorption_coeff": absorption_coeff,
        "gpu_type": gpu_type
    }

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    # Send request
    try:
        response = requests.post(
            f"{API_URL}/simulate",
            json=request_data,
            headers=headers,
            timeout=600  # 10 minute timeout
        )

        response.raise_for_status()  # raises HTTPError if status != 200

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

                print("Insufficient credits for simulation.")
                print("Minimum required: 0.01 credits")
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
        print("The simulation server is taking longer than expected. Please try again.")
        return None

    except requests.exceptions.ConnectionError as e:
        print("Connection failed.")
        print("Unable to reach simulation servers. Please check your network connection and try again.")
        return None

    except requests.exceptions.RequestException as e:
        print("Communication error.")
        print("Unable to process your request at this time. Please try again later.")
        return None

    except ValueError as e:
        print("Invalid server response.")
        print("Received malformed data from server. Our team has been notified.")
        return None


def quick_view_monitors(results: Dict[str, Any], component: str = 'Hz', cmap: str = 'inferno'):
    """Quick visualization of each monitor's first frequency slice.

    Args:
        results: Dictionary from simulate() containing monitor data.
        component: Which field to show:
            - 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz': Individual components
            - '|E|': Electric field magnitude
            - '|H|': Magnetic field magnitude
            - 'all': Total field intensity |E|²+|H|²
        cmap: Colormap to use for visualization (default 'inferno').

    Example:
        >>> results = simulate(...)
        >>> quick_view_monitors(results, 'Hz')  # View Hz component
        >>> quick_view_monitors(results, '|E|')  # View electric field magnitude
        >>> quick_view_monitors(results, 'all')  # View total field intensity
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for visualization")
        return

    for name in results['monitor_data'].keys():
        monitor_data = results['monitor_data'][name]

        # Extract first frequency
        if component in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
            # Single component
            field_map = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}
            comp_idx = field_map[component]
            field_3d = monitor_data[0, comp_idx, :, :, :]
        elif component == '|E|':
            # Electric field magnitude
            E_fields = monitor_data[0, 0:3, :, :, :]
            field_3d = jnp.sqrt(jnp.sum(jnp.abs(E_fields)**2, axis=0))
        elif component == '|H|':
            # Magnetic field magnitude
            H_fields = monitor_data[0, 3:6, :, :, :]
            field_3d = jnp.sqrt(jnp.sum(jnp.abs(H_fields)**2, axis=0))
        elif component == 'all':
            # Total field intensity
            E_fields = monitor_data[0, 0:3, :, :, :]
            H_fields = monitor_data[0, 3:6, :, :, :]
            field_3d = jnp.sqrt(jnp.sum(jnp.abs(E_fields)**2, axis=0) +
                                jnp.sum(jnp.abs(H_fields)**2, axis=0))
        else:
            raise ValueError(f"Unknown component: {component}")

        # Find which dimension is singleton (size 1) or small and average/squeeze
        if field_3d.shape[0] == 1:
            field_2d = field_3d[0, :, :]  # YZ plane
            xlabel, ylabel = 'Y', 'Z'
        elif field_3d.shape[1] == 1:
            field_2d = field_3d[:, 0, :]  # XZ plane
            xlabel, ylabel = 'X', 'Z'
        elif field_3d.shape[2] == 1:
            field_2d = field_3d[:, :, 0]  # XY plane
            xlabel, ylabel = 'X', 'Y'
        else:
            # If no singleton dimension, average across smallest dimension
            min_dim = jnp.argmin(jnp.array(field_3d.shape))
            if min_dim == 0:
                field_2d = jnp.mean(field_3d, axis=0)  # Average across X
                xlabel, ylabel = 'Y', 'Z'
            elif min_dim == 1:
                field_2d = jnp.mean(field_3d, axis=1)  # Average across Y
                xlabel, ylabel = 'X', 'Z'
            else:
                field_2d = jnp.mean(field_3d, axis=2)  # Average across Z
                xlabel, ylabel = 'X', 'Y'

        plt.figure(figsize=(6, 4))

        # For complex fields, take magnitude; for real fields (|E|, |H|, all), use directly
        if jnp.iscomplexobj(field_2d):
            display_field = jnp.abs(field_2d)
            title_prefix = f"|{component}|"
        else:
            display_field = field_2d
            title_prefix = component

        plt.imshow(display_field.T, cmap=cmap, origin='upper', aspect='auto')
        plt.colorbar()
        plt.title(f"{name} - {title_prefix} (freq 0)")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()



