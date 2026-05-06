"""Safe, composable objective functions for inverse design.

Users build loss functions by combining measurement primitives with math
operations. The result is an expression tree (data), not executable code.
This tree is serialized as a JSON-safe spec and sent to the server, where
a fixed JAX interpreter evaluates it. No user code ever runs on the GPU.

Usage:
    import hyperwave_community as hwc
    # or: from hyperwave_community import objectives as obj

    # Simple: maximize mode coupling
    loss = -obj.mode_coupling(mode_field, input_power, P_cross, monitor="wg")

    # Broadband: maximize worst-case across wavelengths
    effs = [obj.mode_coupling(mode, P_in, P_cross, monitor="wg", freq_idx=i)
            for i in range(3)]
    loss = -obj.min_of(*effs)

    # Custom: weighted combination with raw field math
    ey = obj.field("Ey", monitor="focus")
    hz = obj.field("Hz", monitor="focus")
    focusing = obj.sum_spatial(obj.real(ey * obj.conj(hz)))
    loss = -(0.8 * mode_eff + 0.2 * focusing)

    # Evaluate against simulation fields (server-side, JAX)
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(x: Union[Objective, int, float, np.floating]) -> Objective:
    """Wrap a scalar as a Const node, pass Objectives through."""
    if isinstance(x, Objective):
        return x
    if isinstance(x, (int, float, np.floating)):
        return Const(float(x))
    raise TypeError(
        f"Cannot use {type(x).__name__} in objective expression. "
        f"Use Objective nodes or numeric scalars."
    )


# ---------------------------------------------------------------------------
# Serialization context
# ---------------------------------------------------------------------------

class _ArrayRegistry:
    """Collects numpy arrays during serialization, replaces them with indices."""

    def __init__(self):
        self.arrays: List[np.ndarray] = []
        self._seen: Dict[int, int] = {}  # id(arr) -> index (dedup)

    def add(self, arr: np.ndarray) -> int:
        arr_id = id(arr)
        if arr_id in self._seen:
            return self._seen[arr_id]
        idx = len(self.arrays)
        self.arrays.append(arr)
        self._seen[arr_id] = idx
        return idx


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Objective:
    """Base class for objective expression tree nodes.

    Nodes are either measurements (leaf nodes that read from simulation
    fields) or math operations (internal nodes that combine child nodes).
    The tree is pure data. Operator overloading builds the tree, nothing
    is computed until evaluate() is called on the server with JAX.
    """

    def _to_spec(self, registry: _ArrayRegistry) -> dict:
        raise NotImplementedError

    # --- Serialization ---

    def serialize(self) -> Tuple[dict, List[np.ndarray]]:
        """Serialize to a JSON-safe spec dict and a list of numpy arrays.

        Returns:
            (spec, arrays) where spec is a nested dict with array references
            as integer indices, and arrays is the list of referenced arrays.
        """
        registry = _ArrayRegistry()
        spec = self._to_spec(registry)
        return spec, registry.arrays

    # --- Operator overloading ---

    def __neg__(self) -> Objective:
        return Neg(self)

    def __add__(self, other) -> Objective:
        return Add(self, _wrap(other))

    def __radd__(self, other) -> Objective:
        return Add(_wrap(other), self)

    def __sub__(self, other) -> Objective:
        return Sub(self, _wrap(other))

    def __rsub__(self, other) -> Objective:
        return Sub(_wrap(other), self)

    def __mul__(self, other) -> Objective:
        return Mul(self, _wrap(other))

    def __rmul__(self, other) -> Objective:
        return Mul(_wrap(other), self)

    def __truediv__(self, other) -> Objective:
        return Div(self, _wrap(other))

    def __rtruediv__(self, other) -> Objective:
        return Div(_wrap(other), self)

    def __pow__(self, other) -> Objective:
        return Pow(self, _wrap(other))

    def __rpow__(self, other) -> Objective:
        return Pow(_wrap(other), self)

    def __repr__(self) -> str:
        spec, _ = self.serialize()
        return f"Objective({_spec_to_str(spec)})"


def _spec_to_str(spec: dict, depth: int = 0) -> str:
    """Compact string repr of a spec tree."""
    t = spec["type"]
    if t == "const":
        return str(spec["value"])
    if t in ("mode_coupling", "power", "intensity", "field"):
        parts = [f'{t}(monitor="{spec.get("monitor", "?")}"']
        if "component" in spec:
            parts[0] += f', {spec["component"]}'
        return parts[0] + ")"
    if t == "neg":
        return f"-{_spec_to_str(spec['child'])}"
    if t in ("add", "sub", "mul", "div", "pow"):
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}[t]
        return f"({_spec_to_str(spec['left'])} {op} {_spec_to_str(spec['right'])})"
    if t in ("min_of", "max_of"):
        terms = ", ".join(_spec_to_str(s) for s in spec["terms"])
        return f"{t}({terms})"
    if "child" in spec:
        return f"{t}({_spec_to_str(spec['child'])})"
    return t


# ===================================================================
# Leaf nodes: measurements
# ===================================================================

class ModeCoupling(Objective):
    """Mode coupling efficiency via bidirectional overlap integral.

    Computes: |Re(I1 * I2)| / (2 * input_power * mode_cross_power)
    where I1 = sum(E_mode x H_sim*) and I2 = sum(E_sim x H_mode*).
    """

    def __init__(
        self,
        mode_field: np.ndarray,
        input_power: float,
        mode_cross_power: float,
        monitor: str,
        axis: int = 0,
        freq_idx: int = 0,
    ):
        self.mode_field = np.asarray(mode_field)
        self.input_power = float(input_power)
        self.mode_cross_power = float(mode_cross_power)
        self.monitor = monitor
        self.axis = axis
        self.freq_idx = freq_idx

    def _to_spec(self, registry: _ArrayRegistry) -> dict:
        return {
            "type": "mode_coupling",
            "mode_field_idx": registry.add(self.mode_field),
            "input_power": self.input_power,
            "mode_cross_power": self.mode_cross_power,
            "monitor": self.monitor,
            "axis": self.axis,
            "freq_idx": self.freq_idx,
        }


class PowerFlow(Objective):
    """Poynting vector power flow through a monitor plane."""

    def __init__(self, monitor: str, axis: int = 0, freq_idx: int = 0):
        self.monitor = monitor
        self.axis = axis
        self.freq_idx = freq_idx

    def _to_spec(self, registry: _ArrayRegistry) -> dict:
        return {
            "type": "power",
            "monitor": self.monitor,
            "axis": self.axis,
            "freq_idx": self.freq_idx,
        }


class Intensity(Objective):
    """Field intensity |E_component|^2 summed over a monitor."""

    def __init__(self, component: str, monitor: str, freq_idx: int = 0):
        if component not in ("Ex", "Ey", "Ez"):
            raise ValueError(f"component must be Ex, Ey, or Ez, got {component}")
        self.component = component
        self.monitor = monitor
        self.freq_idx = freq_idx

    def _to_spec(self, registry: _ArrayRegistry) -> dict:
        return {
            "type": "intensity",
            "component": self.component,
            "monitor": self.monitor,
            "freq_idx": self.freq_idx,
        }


class FieldComponent(Objective):
    """Raw field component at a monitor. Returns the full array, not a scalar.

    Use with spatial reduction ops (sum_spatial, mean_spatial) and complex
    math (real, imag, conj, abs_val) to build custom measurements.
    """

    _COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    _COMP_INDEX = {c: i for i, c in enumerate(_COMPONENTS)}

    def __init__(self, component: str, monitor: str, freq_idx: int = 0):
        if component not in self._COMPONENTS:
            raise ValueError(
                f"component must be one of {self._COMPONENTS}, got {component}"
            )
        self.component = component
        self.monitor = monitor
        self.freq_idx = freq_idx

    def _to_spec(self, registry: _ArrayRegistry) -> dict:
        return {
            "type": "field",
            "component": self.component,
            "comp_index": self._COMP_INDEX[self.component],
            "monitor": self.monitor,
            "freq_idx": self.freq_idx,
        }


class Const(Objective):
    """Scalar constant."""

    def __init__(self, value: float):
        self.value = float(value)

    def _to_spec(self, registry: _ArrayRegistry) -> dict:
        return {"type": "const", "value": self.value}


# ===================================================================
# Math operation nodes
# ===================================================================

class Neg(Objective):
    def __init__(self, child: Objective):
        self.child = child

    def _to_spec(self, registry: _ArrayRegistry) -> dict:
        return {"type": "neg", "child": self.child._to_spec(registry)}


class _BinaryOp(Objective):
    _type: str = ""

    def __init__(self, left: Objective, right: Objective):
        self.left = left
        self.right = right

    def _to_spec(self, registry: _ArrayRegistry) -> dict:
        return {
            "type": self._type,
            "left": self.left._to_spec(registry),
            "right": self.right._to_spec(registry),
        }


class Add(_BinaryOp):
    _type = "add"


class Sub(_BinaryOp):
    _type = "sub"


class Mul(_BinaryOp):
    _type = "mul"


class Div(_BinaryOp):
    _type = "div"


class Pow(_BinaryOp):
    _type = "pow"


class _UnaryOp(Objective):
    _type: str = ""

    def __init__(self, child: Objective):
        self.child = child

    def _to_spec(self, registry: _ArrayRegistry) -> dict:
        return {"type": self._type, "child": self.child._to_spec(registry)}


class AbsVal(_UnaryOp):
    _type = "abs"


class Log(_UnaryOp):
    _type = "log"


class Log10(_UnaryOp):
    _type = "log10"


class Sqrt(_UnaryOp):
    _type = "sqrt"


class Real(_UnaryOp):
    _type = "real"


class Imag(_UnaryOp):
    _type = "imag"


class Conj(_UnaryOp):
    _type = "conj"


class Relu(_UnaryOp):
    """max(0, x) -- useful for penalty-style objectives."""
    _type = "relu"


class SumSpatial(_UnaryOp):
    """Sum over all spatial dimensions (reduce to scalar)."""
    _type = "sum_spatial"


class MeanSpatial(_UnaryOp):
    """Mean over all spatial dimensions (reduce to scalar)."""
    _type = "mean_spatial"


class _VariadicOp(Objective):
    _type: str = ""

    def __init__(self, terms: Sequence[Objective]):
        if len(terms) < 2:
            raise ValueError(f"{self._type} requires at least 2 terms")
        self.terms = list(terms)

    def _to_spec(self, registry: _ArrayRegistry) -> dict:
        return {
            "type": self._type,
            "terms": [t._to_spec(registry) for t in self.terms],
        }


class MinOf(_VariadicOp):
    """Element-wise minimum across terms. Differentiable via straight-through."""
    _type = "min_of"


class MaxOf(_VariadicOp):
    """Element-wise maximum across terms."""
    _type = "max_of"


# ===================================================================
# Public constructor functions (the user-facing API)
# ===================================================================

def mode_coupling(
    mode_field: np.ndarray,
    input_power: float,
    mode_cross_power: float,
    monitor: str,
    axis: int = 0,
    freq_idx: int = 0,
) -> ModeCoupling:
    """Mode coupling efficiency measurement.

    Args:
        mode_field: Reference mode field, shape (1, 6, ...) or (n_freq, 6, ...).
        input_power: Source input power for normalization.
        mode_cross_power: Mode self-overlap power.
        monitor: Name of the output monitor.
        axis: Propagation axis (0=x, 1=y, 2=z).
        freq_idx: Which frequency index to use.
    """
    return ModeCoupling(mode_field, input_power, mode_cross_power,
                        monitor, axis, freq_idx)


def power(monitor: str, axis: int = 0, freq_idx: int = 0) -> PowerFlow:
    """Poynting vector power flow through a monitor plane.

    Args:
        monitor: Name of the monitor.
        axis: Direction of power flow (0=x, 1=y, 2=z).
        freq_idx: Which frequency index to use.
    """
    return PowerFlow(monitor, axis, freq_idx)


def intensity(component: str, monitor: str, freq_idx: int = 0) -> Intensity:
    """Field intensity |E_component|^2 summed over monitor.

    Args:
        component: "Ex", "Ey", or "Ez".
        monitor: Name of the monitor.
        freq_idx: Which frequency index to use.
    """
    return Intensity(component, monitor, freq_idx)


def field(component: str, monitor: str, freq_idx: int = 0) -> FieldComponent:
    """Raw field component at a monitor.

    Returns the full spatial array, not a scalar. Combine with
    sum_spatial(), real(), conj(), etc. to build custom measurements.

    Args:
        component: "Ex", "Ey", "Ez", "Hx", "Hy", or "Hz".
        monitor: Name of the monitor.
        freq_idx: Which frequency index to use.
    """
    return FieldComponent(component, monitor, freq_idx)


def const(value: float) -> Const:
    """Scalar constant."""
    return Const(value)


# --- Combinators ---

def min_of(*terms: Objective) -> MinOf:
    """Minimum across multiple objectives (worst-case optimization)."""
    return MinOf(list(terms))


def max_of(*terms: Objective) -> MaxOf:
    """Maximum across multiple objectives."""
    return MaxOf(list(terms))


def abs_val(x: Objective) -> AbsVal:
    """Absolute value."""
    return AbsVal(x)


def log(x: Objective) -> Log:
    """Natural logarithm."""
    return Log(x)


def log10(x: Objective) -> Log10:
    """Base-10 logarithm."""
    return Log10(x)


def sqrt(x: Objective) -> Sqrt:
    """Square root."""
    return Sqrt(x)


def real(x: Objective) -> Real:
    """Real part of complex value."""
    return Real(x)


def imag(x: Objective) -> Imag:
    """Imaginary part of complex value."""
    return Imag(x)


def conj(x: Objective) -> Conj:
    """Complex conjugate."""
    return Conj(x)


def relu(x: Objective) -> Relu:
    """max(0, x). Useful for penalty terms: relu(threshold - measurement)."""
    return Relu(x)


def sum_spatial(x: Objective) -> SumSpatial:
    """Sum over all spatial dimensions, reducing to a scalar."""
    return SumSpatial(x)


def mean_spatial(x: Objective) -> MeanSpatial:
    """Mean over all spatial dimensions, reducing to a scalar."""
    return MeanSpatial(x)

