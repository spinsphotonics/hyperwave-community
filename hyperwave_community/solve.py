"""Minimal FDTD solver for mode_converter.

Copied from hyperwave/solve.py -- contains only the functions needed by
mode_converter (mem_efficient_multi_freq and its dependencies).
"""

from functools import partial
from typing import NamedTuple, List

import jax
import jax.numpy as jnp


class FreqBand(NamedTuple):
    """Describes `num` regularly spaced values within `[start, stop]`."""

    start: float
    stop: float
    num: int

    @property
    def values(self) -> jax.Array:
        if self.num == 1:
            return jnp.array([(self.start + self.stop) / 2])
        else:
            return jnp.linspace(self.start, self.stop, self.num)


def mem_efficient_multi_freq(
    freq_band: tuple[float, float, int],
    permittivity: jax.Array,
    conductivity: jax.Array,
    source_field: jax.Array,
    source_offset: tuple[int, int, int],
    monitors: list,
    source_ramp_periods: float = 10.0,
    max_steps: int = 5_000,
    check_every_n: int = 200,
    max_courant_factor: float = 0.99,
    convergence_threshold: float = 1e-6,
) -> tuple[list, list[int], list[jax.Array]]:
    """Memory-efficient multi-frequency solver that stores multiple monitor volumes."""
    if not monitors:
        raise ValueError("At least one monitor must be provided")

    freq_band = FreqBand(*freq_band)
    dt, sample_every_n = _sampling_strategy(freq_band, permittivity, max_courant_factor)
    run_every_n = max(check_every_n, sample_every_n * 2 * freq_band.num)

    output_steps = list(
        range(
            run_every_n - sample_every_n * (2 * freq_band.num - 1),
            run_every_n + 1,
            sample_every_n,
        )
    )

    max_steps = ((max_steps + run_every_n) // run_every_n) * run_every_n

    t = dt * jnp.arange(0, max_steps)
    e_wave = _ramped_sinusoids(
        t=t + dt / 2, freq_band=freq_band, num_ramp_periods=source_ramp_periods
    )
    h_wave = _ramped_sinusoids(
        t=t, freq_band=freq_band, num_ramp_periods=source_ramp_periods
    )
    source_waveform = jnp.concatenate(
        [
            jnp.broadcast_to(e_wave[:, None, :], (e_wave.shape[0], 3, e_wave.shape[1])),
            jnp.broadcast_to(h_wave[:, None, :], (h_wave.shape[0], 3, h_wave.shape[1])),
        ],
        axis=1,
    )

    full_shape = permittivity.shape[1:]
    field = jnp.zeros((6,) + full_shape)

    output_shapes = [monitor.shape for monitor in monitors]
    output_offsets = [monitor.offset for monitor in monitors]

    errs = []
    steps = []
    previous_freq_out = None

    for start_step in range(0, max_steps, run_every_n):
        field, outs = time_domain(
            dt=dt,
            permittivity=permittivity,
            conductivity=conductivity,
            source_field=source_field,
            source_waveform=source_waveform[..., start_step : start_step + run_every_n],
            source_offset=source_offset,
            output_shapes=output_shapes,
            output_offsets=output_offsets,
            output_steps=output_steps,
            field=field,
        )

        output_times = dt * start_step + t[jnp.array(output_steps)]
        current_freq_out = [
            jnp.concatenate(
                [
                    _project(
                        field=out[:, :3, ...], freq_band=freq_band, t=output_times
                    ),
                    _project(
                        field=out[:, 3:, ...],
                        freq_band=freq_band,
                        t=output_times - dt / 2,
                    ),
                ],
                axis=1,
            )
            for out in outs
        ]

        if previous_freq_out is not None:
            err = _monitor_convergence_error(
                current_fields=current_freq_out[0],
                previous_fields=previous_freq_out[0],
            )
            errs.append(err)
            steps.append(start_step + run_every_n)

            if jnp.max(err) < convergence_threshold:
                print(f"Converged at step {start_step + run_every_n}")
                break

        previous_freq_out = current_freq_out

    return current_freq_out, steps, errs


def time_domain(
    dt: float,
    permittivity: jax.Array,
    conductivity: jax.Array,
    source_field: jax.Array,
    source_waveform: jax.Array,
    source_offset: tuple[int, int, int],
    output_shapes: list[tuple[int, int, int]],
    output_offsets: list[tuple[int, int, int]],
    output_steps: list[int],
    field: jax.Array,
) -> tuple[jax.Array, list[jax.Array]]:
    """Execute the finite-difference time-domain (FDTD) simulation method."""
    z = conductivity * dt / (2 * permittivity)
    ca = (1 - z) / (1 + z)
    cb = dt / permittivity / (1 + z)

    def step_fn(step: int, field: jax.Array) -> jax.Array:
        src = jnp.real(
            jnp.sum(
                jnp.expand_dims(source_waveform[:, :, step], (-3, -2, -1))
                * source_field,
                axis=0,
            )
        )
        return _fdtd_update(
            field=field,
            source=src,
            offset=source_offset,
            dt=dt,
            ca=ca,
            cb=cb,
        )

    outs = []
    for start, end in zip([0] + output_steps[:-1], output_steps):
        field = jax.lax.fori_loop(
            lower=start,
            upper=end,
            body_fun=step_fn,
            init_val=field,
        )
        outs.append(
            [
                _get(field, shape, offset)
                for shape, offset in zip(output_shapes, output_offsets)
            ]
        )

    return field, [jnp.stack(out) for out in zip(*outs)]


def _fdtd_update(
    field: jax.Array,
    source: jax.Array,
    offset: tuple[int, int, int],
    dt: float,
    ca: jax.Array,
    cb: jax.Array,
) -> jax.Array:
    """A single step of the FDTD update of `field`."""
    e, h = _eh_split(field)
    e_src, h_src = _eh_split(source)

    h = h - dt * _curl_and_source(field=e, source=h_src, offset=offset, is_forward=True)
    e = ca * e + cb * _curl_and_source(
        field=h, source=e_src, offset=offset, is_forward=False
    )

    return _eh_join(e, h)


def _curl_and_source(
    field: jax.Array, source: jax.Array, offset: tuple[int, int, int], is_forward: bool
) -> jax.Array:
    """Take the curl of `field` and adds `source` at `offset`."""
    return _at(
        field=_curl(field, is_forward), shape=source.shape[-3:], offset=offset
    ).add(source)


def _curl(field: jax.Array, is_forward: bool) -> jax.Array:
    """Computes the curl of `field` for forward- and backward-differences."""
    fx, fy, fz = [field[..., i, :, :, :] for i in range(3)]
    dx, dy, dz = [
        partial(
            _spatial_diff,
            axis=axis,
            is_forward=is_forward,
        )
        for axis in range(-3, 0)
    ]
    return jnp.stack([dy(fz) - dz(fy), dz(fx) - dx(fz), dx(fy) - dy(fx)], axis=-4)


def _spatial_diff(field: jax.Array, axis: int, is_forward: bool) -> jax.Array:
    """Computes the spatial difference of `field` along `axis`."""
    if is_forward:
        return jnp.roll(field, shift=-1, axis=axis) - field
    return field - jnp.roll(field, shift=+1, axis=axis)


def _at(field: jax.Array, shape: tuple[int, int, int], offset: tuple[int, int, int]):
    """Modify `shape` values of `field` at `offset`."""
    return field.at[
        ...,
        offset[0] : offset[0] + shape[0],
        offset[1] : offset[1] + shape[1],
        offset[2] : offset[2] + shape[2],
    ]


def _get(
    field: jax.Array, shape: tuple[int, int, int], offset: tuple[int, int, int]
) -> jax.Array:
    """Returns `shape` values of `field` at `offset`."""
    return field[
        ...,
        offset[0] : offset[0] + shape[0],
        offset[1] : offset[1] + shape[1],
        offset[2] : offset[2] + shape[2],
    ]


def _eh_split(array: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split the `(6, ...)`-array into electric and magnetic components."""
    return tuple(jnp.split(array, indices_or_sections=2))


def _eh_join(e: jax.Array, h: jax.Array) -> jax.Array:
    """Join the electric and magnetic components into a `(6, ...)`-array."""
    return jnp.concatenate((e, h), axis=0)


def _project(
    field: jax.Array,
    freq_band: FreqBand,
    t: jax.Array,
) -> jax.Array:
    """Projects `snapshots` at `t` to `freq_band` frequencies."""
    wt = freq_band.values[None, :] * t[:, None]
    P = jnp.concatenate([jnp.cos(wt), -jnp.sin(wt)], axis=1)
    res = jnp.einsum("ij,j...->i...", jnp.linalg.inv(P), field)
    return res[: freq_band.num] + 1j * res[freq_band.num :]


def _monitor_convergence_error(
    current_fields: jax.Array,
    previous_fields: jax.Array,
) -> jax.Array:
    """Compute convergence error based on field stability in monitor volume."""
    current_magnitude = jnp.sqrt(jnp.sum(jnp.abs(current_fields)**2, axis=1))
    previous_magnitude = jnp.sqrt(jnp.sum(jnp.abs(previous_fields)**2, axis=1))
    relative_change = jnp.abs(current_magnitude - previous_magnitude) / (previous_magnitude + 1e-12)
    return jnp.mean(relative_change, axis=(-3, -2, -1))


def _sampling_strategy(
    freq_band: FreqBand,
    permittivity: jax.Array,
    max_courant_factor: float,
) -> tuple[float, int]:
    """Carefully chooses `(dt, sample_every_n)` simulation parameters."""
    sampling_interval = _sampling_interval(freq_band)
    dt = max_courant_factor * jnp.sqrt(jnp.min(permittivity)) / jnp.sqrt(3)

    if freq_band.num == 1:
        sample_every_n = int(round(sampling_interval / dt))
    else:
        n = int(jnp.floor(sampling_interval / dt))
        dt = sampling_interval / (n + 1)
        sample_every_n = n + 1

    return dt, sample_every_n


def _sampling_interval(freq_band: FreqBand) -> float:
    """Snapshot interval for efficiently sampling `freq_band` frequencies."""
    w = freq_band.values
    if len(w) == 1:
        return float(jnp.pi / (2 * w[0]))
    else:
        w_avg = (w[0] + w[-1]) / 2
        dw = abs(w[-1] - w[0]) / (len(w) - 1)
        return _round_to_mult(
            2 * jnp.pi / (len(w) * dw),
            multiple=jnp.pi / (len(w) * w_avg),
            offset=0.5,
        )


def _round_to_mult(x, multiple, offset=0):
    """Rounds `x` to the nearest multiple of `multiple`."""
    return (round(x / multiple - offset) + offset) * multiple


def _ramped_sinusoids(
    t: jax.Array,
    freq_band: FreqBand,
    num_ramp_periods: float,
):
    """Sinusoid that ramps to steady-state over `num_ramp_periods`."""
    source_ramp_time = 2 * jnp.pi * num_ramp_periods / freq_band.values[0]
    return jnp.where(
        t < source_ramp_time,
        0.5 * (1 - jnp.cos(jnp.pi * t / source_ramp_time)),
        1.0,
    ) * jnp.exp(1j * freq_band.values[:, None] * t)
