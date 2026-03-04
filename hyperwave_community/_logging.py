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
