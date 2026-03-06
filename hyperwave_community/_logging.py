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


def set_device(mode: str = "auto"):
    """Configure JAX compute device.

    Call this before any JAX operations (e.g. before creating structures).

    Args:
        mode: ``"auto"`` (default) uses GPU if available, falls back to CPU.
            ``"cpu"`` forces CPU-only. ``"gpu"`` forces GPU (raises if
            unavailable).

    Raises:
        RuntimeError: If ``mode="gpu"`` but no GPU is found.
        ValueError: If *mode* is not one of ``"auto"``, ``"cpu"``, ``"gpu"``.
    """
    import jax

    mode = mode.lower().strip()
    if mode not in ("auto", "cpu", "gpu"):
        raise ValueError(f"mode must be 'auto', 'cpu', or 'gpu', got '{mode}'")

    if mode == "cpu":
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        logger.info("Device: CPU (forced)")
        return

    gpu_devices = jax.devices("gpu") if _has_gpu() else []

    if mode == "gpu":
        if not gpu_devices:
            raise RuntimeError(
                "mode='gpu' but no GPU found. Install jaxlib with CUDA support "
                "or use mode='auto' for CPU fallback."
            )
        jax.config.update("jax_default_device", gpu_devices[0])
        logger.info(f"Device: GPU ({gpu_devices[0]})")
        return

    # auto
    if gpu_devices:
        jax.config.update("jax_default_device", gpu_devices[0])
        logger.info(f"Device: GPU ({gpu_devices[0]})")
    else:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        logger.info("Device: CPU (no GPU found)")


def _has_gpu() -> bool:
    """Check if JAX can see any GPU devices."""
    try:
        import jax
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False
