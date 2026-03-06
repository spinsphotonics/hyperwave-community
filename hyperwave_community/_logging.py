import logging

logger = logging.getLogger("hyperwave")
logger.setLevel(logging.WARNING)

_formatter = logging.Formatter("%(message)s")
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.propagate = False


def set_verbose(verbose=True):
    """Enable or disable verbose output.

    When enabled, shows progress messages during simulation,
    optimization, and other long-running operations.
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)


def set_debug(debug=True):
    """Enable or disable debug-level output.

    Shows detailed internal state for troubleshooting.
    """
    logger.setLevel(logging.DEBUG if debug else logging.WARNING)


def _has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is physically present via nvidia-smi."""
    import subprocess
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _ensure_jax_installed(need_gpu: bool):
    """Install the correct jaxlib variant if JAX cannot see the expected device."""
    import subprocess, sys

    try:
        import jax
        if need_gpu:
            try:
                if jax.devices("gpu"):
                    return
            except RuntimeError:
                pass
        else:
            return
    except ImportError:
        pass

    package = "jax[cuda12]" if need_gpu else "jax[cpu]"
    logger.info("Installing %s...", package)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", package],
        stdout=subprocess.DEVNULL,
    )
    import importlib
    if "jax" in sys.modules:
        importlib.reload(sys.modules["jax"])


def set_device(mode: str = "auto"):
    """Configure JAX compute device.

    Ensures the correct ``jaxlib`` variant is installed, then sets the
    default JAX device. Call before any JAX operations.

    Args:
        mode: ``"auto"`` (default) detects GPU via ``nvidia-smi`` and
            installs/uses CUDA jaxlib when available, CPU otherwise.
            ``"cpu"`` forces CPU-only. ``"gpu"`` forces GPU (raises if
            no NVIDIA GPU is found).

    Raises:
        RuntimeError: If ``mode="gpu"`` but no NVIDIA GPU is present.
        ValueError: If *mode* is not one of ``"auto"``, ``"cpu"``, ``"gpu"``.
    """
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="An NVIDIA GPU may be present.*CUDA-enabled jaxlib is not installed",
        module="jax",
    )

    mode = mode.lower().strip()
    if mode not in ("auto", "cpu", "gpu"):
        raise ValueError(f"mode must be 'auto', 'cpu', or 'gpu', got '{mode}'")

    has_gpu = _has_nvidia_gpu()

    if mode == "gpu" and not has_gpu:
        raise RuntimeError(
            "mode='gpu' but no NVIDIA GPU found (nvidia-smi failed)."
        )

    need_gpu = (mode == "gpu") or (mode == "auto" and has_gpu)
    _ensure_jax_installed(need_gpu)

    import jax

    if need_gpu:
        gpu_devices = jax.devices("gpu")
        jax.config.update("jax_default_device", gpu_devices[0])
        logger.info("Device: GPU (%s)", gpu_devices[0])
    else:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        logger.info("Device: CPU")
