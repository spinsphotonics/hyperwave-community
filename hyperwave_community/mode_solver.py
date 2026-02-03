"""Waveguide eigenmode solver for hyperwave-community.

This module provides the mode solver for computing waveguide eigenmodes.
It uses JAX's LOBPCG algorithm to solve the eigenvalue problem.
"""

from functools import partial
from math import prod
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard


class FreqBand(NamedTuple):
    """Describes `num` regularly spaced values within `[start, stop]`.

    The convention for `num=1` is that the FreqBand
    represents the single-element array with value `(start + stop) / 2`.

    Args:
        start: Extremal value of the band.
        stop: Other extremal value of the band.
        num: Number of equally-spaced values within `[start, stop]`.
    """

    start: float
    stop: float
    num: int

    @property
    def values(self) -> jax.Array:
        """Values represented by band."""
        if self.num == 1:
            return jnp.array([(self.start + self.stop) / 2])
        else:
            return jnp.linspace(self.start, self.stop, self.num)


def _spatial_diff(field: jax.Array, axis: int, is_forward: bool) -> jax.Array:
    """Computes the spatial difference of `field` along `axis`."""
    if is_forward:
        return jnp.roll(field, shift=-1, axis=axis) - field
    return field - jnp.roll(field, shift=+1, axis=axis)


def _wg_operator(omega: float, permittivity: jax.Array, axis: int):
    """Waveguide operator for mode eigenvalue problem.

    Specifically, we solve the waveguide problem for the H-fields. These, in
    turn, are the values which are needed for the E-field source.

    Args:
        omega: Angular frequency to solve for.
        permittivity: `(3, xx, yy, zz)` array of permittivity values.
        axis: `0`, `1`, or `2` corresponding to x-, y-, or z-axis.

    Returns:
        Waveguide operator that can be used to solve the eigenvalue problem
        iteratively.
    """
    shape = permittivity.shape[1:]

    dfi, dbi, dfj, dbj = [
        partial(
            _spatial_diff,
            axis=((axis + axis_shift) % 3) - 3,
            is_forward=is_forward,
        )
        for (axis_shift, is_forward) in ((1, True), (1, False), (2, True), (2, False))
    ]

    def _split(u):
        return jnp.split(u, indices_or_sections=2, axis=1)

    def _concat(u):
        return jnp.concatenate(u, axis=1)

    def curl_to_k(u):
        ui, uj = _split(u)
        return dbi(uj) - dbj(ui)

    def curl_to_ij(u):
        return _concat([-dfj(u), dfi(u)])

    def div(u):
        ui, uj = _split(u)
        return dfi(ui) + dfj(uj)

    def grad(u):
        return _concat([dbi(u), dbj(u)])

    ei, ej, ek = tuple(permittivity[(i + 1) % 3] for i in range(3))
    eji = jnp.stack([ej, ei], axis=0)

    def op(x):
        u = jnp.reshape(x.T, (-1, 2) + shape)
        return jnp.reshape(
            omega**2 * eji * u + eji * curl_to_ij(curl_to_k(u) / ek) + grad(div(u)),
            x.shape[::-1],
        ).T

    return op


def mode(
    freq_band: Tuple[float, float, int],
    permittivity: jax.Array,
    axis: int,
    mode_num: int,
    random_seed: int = 0,
    min_modes_in_solve: int = 10,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Solve for first `num_modes` propagating modes.

    Args:
        freq_band: `(start, stop, num)` of frequency values.
        permittivity: `(3, xx, yy, zz)` array of real-valued permittivity
          values.
        axis: One of `0`, `1`, or `2` denoting the x-, y-, or z-propagation
            axes respectively.
        mode_num: Mode number to solve for (where the fundamental mode
            corresponds to `0`).
        random_seed: Defaults to `0`.
        min_modes_in_solve: Solving the eigenvalue problem must include at least
            `min_modes_in_solve` modes.

    Returns:
        field: `(num, 6, xx, yy, zz)` array of the real-valued mode patterns.
        beta: `(num,)` array of wavevectors.
        errs: `(num,)` array of error values.
    """
    freq_band_obj = FreqBand(*freq_band)
    field_shape = permittivity.shape[1:]
    shape = (2 * prod(field_shape), min_modes_in_solve)

    # Solve for either `min_modes_in_solve` modes, or else twice as many modes
    # as needed, whichever is greater. Solving for more modes than needed
    # seems to improve accuracy.
    min_modes_in_solve = max(min_modes_in_solve, (mode_num + 1) * 2)

    # Initial value to use when solving the eigenvalue problem.
    x = jax.random.normal(jax.random.PRNGKey(random_seed), shape)

    # Solve a separate eigenvalue problem for each frequency.
    errs, modes, betas = [], [], []
    for omega in freq_band_obj.values:
        op = _wg_operator(omega, permittivity, axis)
        betas_squared, u, _ = lobpcg_standard(op, x)
        err = jnp.linalg.norm(op(u) - betas_squared * u, axis=0)
        mode_result = jnp.reshape(u.T, (-1, 2) + field_shape)

        errs.append(err[mode_num])
        modes.append(mode_result[mode_num])
        betas.append(jnp.sqrt(betas_squared[mode_num]))

    errs = jnp.stack(errs)
    betas = jnp.stack(betas)
    modes = jnp.stack(modes)

    # Assign mode patterns to correct field components.
    fields = jnp.zeros((modes.shape[0], 3) + modes.shape[-3:])
    if axis == 0:
        fields = fields.at[:, 1].set(modes[:, 1])
        fields = fields.at[:, 2].set(modes[:, 0])
    elif axis == 1:
        fields = fields.at[:, 0].set(modes[:, 1])
        fields = fields.at[:, 2].set(modes[:, 0])
    elif axis == 2:
        fields = fields.at[:, 0].set(modes[:, 1])
        fields = fields.at[:, 1].set(modes[:, 0])

    return fields, betas, errs
