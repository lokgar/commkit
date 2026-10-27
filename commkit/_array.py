"""Array shape and validation helpers (private).

Indexing, broadcasting and coercion operations on NumPy/CuPy arrays, used by
every module: :func:`validate_array`, the SISO/MIMO shape pair
:func:`as_2d` / :func:`restore_1d`, :func:`broadcast_channels`,
:func:`require_channels`, :func:`to_report_scalar` (host values of the
metrics and summaries) and :func:`warn_if_double` (the
:class:`~commkit.PrecisionWarning` of the fixed-precision blocks).  This
module imports only ``backend``: nothing from ``core``, DSP modules or
plotting.
"""

from typing import Any, overload

import numpy as np

from .backend import (
    ArrayType,
    _is_cupy_array,
    _reject_foreign_array,
    get_array_module,
    to_device,
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_array(
    v: Any, name: str = "array", complex_only: bool = False
) -> ArrayType:
    """Validate and coerce input data into a numeric array.

    Existing NumPy or CuPy arrays are passed through unchanged (preserving
    device placement). All other inputs (Python scalars, lists, tuples) are
    coerced to NumPy via ``np.asarray``; there is no automatic promotion to
    CuPy for non-array inputs. Optionally enforces complex-valued dtype.

    Parameters
    ----------
    v : array_like or any
        Input data to validate.
    name : str, default "array"
        Variable name used in error messages.
    complex_only : bool, default False
        If True, ensures the resulting array is complex-valued.

    Returns
    -------
    ndarray
        NumPy or CuPy array (CuPy only when ``v`` was already a CuPy array).

    Raises
    ------
    ValueError
        If the input cannot be converted to a supported array type.
    """
    if v is None:
        return None

    # Coerce lists/tuples and scalars to NumPy; arrays from other frameworks
    # (JAX, PyTorch, ...) are rejected rather than silently copied.
    if not (isinstance(v, np.ndarray) or _is_cupy_array(v)):
        _reject_foreign_array(v)
        try:
            v = np.asarray(v)
        except Exception as err:
            raise ValueError(
                f"Could not convert {name} of type {type(v)} to array."
            ) from err

    # Ensure it's a numeric array (not object, string, etc.)
    if v.dtype.kind not in "biufc":
        raise ValueError(
            f"Expected numeric array for {name}, got dtype {v.dtype} (kind {v.dtype.kind})"
        )

    if complex_only and not np.iscomplexobj(v):
        xp = get_array_module(v)
        # Preserve single-precision: float32 -> complex64, everything else -> complex128
        complex_dtype = xp.complex64 if v.dtype == xp.float32 else xp.complex128
        v = v.astype(complex_dtype)

    return v


# ---------------------------------------------------------------------------
# Array shape helpers
# ---------------------------------------------------------------------------
#
# CommKit's SISO/MIMO convention is ``(N,)`` / ``(C, N)`` with time on the last
# axis.  Nearly every DSP entry point therefore
# promotes a 1-D input to ``(1, N)``, runs one vectorized channel-batched
# implementation, and squeezes the leading axis back off on the way out.  These
# helpers are that idiom, defined once, so the promotion is validated the same
# way everywhere instead of silently passing 3-D (or 0-d) input through to a
# confusing downstream broadcast error.
#
# They are pure indexing/broadcast operations and therefore correct on every
# array type the library sees - NumPy and CuPy - without dispatching.


def as_2d(arr: ArrayType, *, name: str = "array") -> tuple[ArrayType, bool]:
    """Promote a SISO ``(N,)`` array to the MIMO layout ``(1, N)``.

    The canonical entry half of the library's SISO/MIMO shape idiom; pair it
    with :func:`restore_1d` to squeeze the promoted axis back off the outputs.

    Parameters
    ----------
    arr : array_like
        Input array, ``(N,)`` (SISO) or ``(C, N)`` (MIMO, time last).
        Must already be an array (call ``dispatch`` first); no conversion or
        host transfer is performed.
    name : str, default "array"
        Variable name used in the error message.

    Returns
    -------
    arr_2d : ndarray
        ``arr[None, :]`` for 1-D input, ``arr`` itself (no copy) for 2-D.
    was_1d : bool
        Whether the promotion happened; pass it to :func:`restore_1d`.

    Raises
    ------
    ValueError
        If ``arr`` is 0-d or has more than two dimensions.  CommKit signals
        carry at most a channel axis and a time axis; a 3-D input is a caller
        error, not a batch dimension.
    """
    ndim = np.ndim(arr)  # reads ``arr.ndim``; never converts CuPy to host
    if ndim == 1:
        return arr[None, :], True
    if ndim == 2:
        return arr, False
    raise ValueError(
        f"{name} must be 1-D (N,) for SISO or 2-D (C, N) for MIMO with time on "
        f"the last axis; got ndim={ndim}."
    )


@overload
def restore_1d(was_1d: bool, arr: ArrayType, /) -> ArrayType: ...


@overload
def restore_1d(
    was_1d: bool, arr: ArrayType, arr2: ArrayType, /, *rest: ArrayType
) -> tuple[ArrayType, ...]: ...


def restore_1d(was_1d: bool, *arrays: ArrayType) -> ArrayType | tuple[ArrayType, ...]:
    """Undoes :func:`as_2d` on one or more outputs.

    Parameters
    ----------
    was_1d : bool
        The flag returned by :func:`as_2d`.
    *arrays : array_like
        Channel-batched results, each ``(1, ...)`` when ``was_1d`` is True.

    Returns
    -------
    ndarray or tuple of ndarray
        ``arr[0]`` per input when ``was_1d``, otherwise the inputs unchanged.
        A single input returns bare (not a 1-tuple), so both
        ``out = restore_1d(was_1d, out)`` and
        ``drift, pn = restore_1d(was_1d, drift, pn)`` read naturally.
    """
    if not arrays:
        raise ValueError("restore_1d() requires at least one array.")
    out = tuple(a[0] for a in arrays) if was_1d else arrays
    return out[0] if len(out) == 1 else out


def broadcast_channels(
    ref: ArrayType, num_channels: int, xp: Any = None, *, name: str = "reference"
) -> ArrayType:
    """Broadcast a shared reference sequence across ``num_channels`` channels.

    Replaces the ad-hoc ``if ref.ndim == 1: ref = ref[None, :]`` promotion used
    at every reference/pilot input, adding the channel-count check that the
    bare promotion leaves to a later, far more opaque broadcast failure.

    Parameters
    ----------
    ref : array_like
        Reference sequence: ``(L,)`` shared by all channels, ``(1, L)``
        (broadcast), or ``(C, L)`` (per-channel).
    num_channels : int
        Number of channels ``C`` the reference must cover.
    xp : module, optional
        Array module to broadcast with.  Inferred from ``ref`` when omitted.
    name : str, default "reference"
        Variable name used in the error message.

    Returns
    -------
    ndarray
        A ``(C, L)`` view.  Shared references are returned as a **read-only
        broadcast view** (no data is copied); call ``.copy()`` before writing.

    Raises
    ------
    ValueError
        If ``ref`` is not 1-D/2-D, or its channel count is neither
        ``num_channels`` nor 1.
    """
    if xp is None:
        xp = get_array_module(ref)
    ndim = np.ndim(ref)
    if ndim == 1:
        return xp.broadcast_to(ref[None, :], (num_channels, ref.shape[-1]))
    if ndim == 2:
        c = ref.shape[0]
        if c == num_channels:
            return ref
        if c == 1:
            return xp.broadcast_to(ref, (num_channels, ref.shape[-1]))
        raise ValueError(
            f"{name} has {c} channels, which matches neither the signal's "
            f"{num_channels} channels nor 1 (broadcast)."
        )
    raise ValueError(f"{name} must be 1-D (L,) or 2-D (C, L); got ndim={ndim}.")


def require_channels(
    arr: ArrayType,
    num_channels: int,
    *,
    name: str = "samples",
    description: str | None = None,
) -> ArrayType:
    """Validate a strict MIMO layout with an exact channel count.

    For entry points defined only for a fixed number of channels
    (dual-polarization channel models and polarization equalizers), where a
    SISO input is a caller error rather than something to promote.

    Parameters
    ----------
    arr : array_like
        Input array; must be ``(num_channels, N)``.
    num_channels : int
        Required channel count (e.g. ``2`` for dual-pol).
    name : str, default "samples"
        Variable name used in the error message.
    description : str, optional
        Domain wording for the requirement, e.g. ``"dual-pol input with shape
        (2, N)"``.  Defaults to a generic ``"a 2-D (C, N) array"``.

    Returns
    -------
    ndarray
        ``arr`` unchanged.

    Raises
    ------
    ValueError
        If ``arr`` is not 2-D or does not have exactly ``num_channels`` rows.
    """
    ndim = np.ndim(arr)
    if ndim != 2 or arr.shape[0] != num_channels:
        shape = tuple(arr.shape) if hasattr(arr, "shape") else np.shape(arr)
        what = description or f"a 2-D ({num_channels}, N) array"
        raise ValueError(
            f"{name} must be {what} with time on the last axis; got shape {shape}."
        )
    return arr


def to_report_scalar(values: Any, *, was_1d: bool | None = None) -> float | np.ndarray:
    """A reporting-layer result on the host: a Python float or a float64 array.

    The reporting-layer counterpart of :func:`restore_1d`: channel-batched
    results are computed as ``(C,)`` vectors, but ``(N,)`` input reports a
    plain float.  Device arrays are transferred to the host internally, so
    this is safe to call on a CuPy result directly.

    Parameters
    ----------
    values : array_like or scalar
        Per-channel result, ``(C,)``, or a 0-d value.
    was_1d : bool, optional
        Whether the input was ``(N,)``.  ``True`` collapses the single
        channel to a float; ``False`` keeps the array, so ``(1, N)`` input
        reports ``(1,)``.  ``None`` collapses only a 0-d value, for results
        already reduced by input rank.

    Returns
    -------
    float or numpy.ndarray
        A Python float for ``(N,)`` input, otherwise a ``float64`` NumPy array.
    """
    arr = np.asarray(to_device(values, "cpu"), dtype=np.float64)
    if was_1d is None:
        was_1d = arr.ndim == 0
    if not was_1d:
        return arr
    if arr.size != 1:
        raise ValueError(f"(N,) input reports one value, got shape {arr.shape}.")
    return float(arr.reshape(-1)[0])


def warn_if_double(arr: Any, function_name: str, working: str = "complex64") -> None:
    """Warn that a single-precision block received double-precision data.

    Emits :class:`~commkit.backend.PrecisionWarning` when ``arr`` is
    complex128 or float64 and ``function_name`` computes in the
    single-precision ``working`` dtype, attributed to the caller's line.
    """
    dtype = getattr(arr, "dtype", None)
    if dtype is None or dtype not in (np.complex128, np.float64):
        return
    import sys
    import warnings

    from .backend import PrecisionWarning

    # Point the warning at the first frame outside commkit.
    frame: Any = sys._getframe(1)  # the caller: stacklevel 2
    level = 2
    while frame is not None and frame.f_globals.get("__name__", "").startswith(
        "commkit"
    ):
        frame, level = frame.f_back, level + 1
    warnings.warn(
        f"{function_name} computes in {working}: the {dtype} input is reduced "
        "to single precision.",
        PrecisionWarning,
        stacklevel=level,
    )
