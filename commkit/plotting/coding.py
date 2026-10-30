"""Channel-coding plots.

:func:`plot_llr_distribution` draws the conditional distributions of LLRs
given the transmitted bits and checks their consistency;
:func:`plot_parity_check` draws the structure and node degrees of a
parity-check matrix; :func:`plot_exit_chart` draws an
:class:`~commkit.coding.ExitChart`.

LLRs follow the library's convention, ``log P(b = 0 | y) / P(b = 1 | y)``:
positive means bit 0.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .._array import validate_array
from ..backend import dispatch, to_device
from ..logger import logger
from .theme import _finish, _get_axis, _grid_figsize, _square_figsize

__all__ = ["plot_exit_chart", "plot_llr_distribution", "plot_parity_check"]

# Counts per histogram bin below which the consistency panel skips the bin.
_MIN_COUNT = 20

# Matrices up to this many entries are drawn as an image, larger ones as one
# marker per nonzero entry.
_DENSE_ENTRIES = 1_000_000


def plot_llr_distribution(
    llrs: Any,
    bits: Any,
    *,
    bits_per_symbol: int | None = None,
    bins: int = 100,
    ax: Any | None = None,
    title: str | None = None,
    show: bool = False,
) -> tuple[Any, Any] | None:
    """Plot the distributions of LLRs given the transmitted bits, and their consistency.

    Panels: the densities ``p(L | b = 0)`` and ``p(L | b = 1)``, and the
    consistency check ``log p(L | b = 0) / p(L | b = 1)`` against ``L``.
    LLRs that are true log-likelihood ratios lie on the diagonal; a slope
    below 1 means overconfident LLRs (a ``noise_var`` too small, for
    example), above 1 underconfident ones, and a curve that bends away
    from the diagonal a mismatched demapper model.  All channels are pooled.

    Parameters
    ----------
    llrs : array_like
        Real LLRs, ``(N,)`` or ``(C, N)``, positive for bit 0, as
        :func:`commkit.mapping.compute_llr` returns.
    bits : array_like
        The transmitted bits, 0 or 1, of the same shape.
    bits_per_symbol : int, optional
        Bits per symbol label: the consistency panel then draws one curve per
        bit position of the label, which differ for the levels of a QAM
        label.  ``None`` pools all bits.  ``N`` must be a multiple of it.
    bins : int, default 100
        Histogram bins over the central 99.9 % of ``|L|``.
    ax : array_like of matplotlib.axes.Axes, optional
        Two axes to draw on; ``None`` creates a figure.
    title : str, optional
        Title of the figure.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.

    Raises
    ------
    ValueError
        If the shapes differ, ``bits`` holds values other than 0 and 1 or
        only one of them, or ``N`` is not a multiple of ``bits_per_symbol``.
    TypeError
        If ``llrs`` is complex or ``bits`` is not integer or bool.

    Notes
    -----
    The histograms are computed on the input's device; only the bin counts
    are transferred.  The consistency panel skips bins with fewer than
    20 LLRs of either bit value.
    """
    x, xp, _ = dispatch(validate_array(llrs, name="llrs"))
    b = validate_array(bits, name="bits")
    if xp.iscomplexobj(x):
        raise TypeError("plot_llr_distribution(): llrs must be real.")
    if b.dtype.kind not in "biu":
        raise TypeError(
            f"plot_llr_distribution(): bits must be integers or bool, got {b.dtype}."
        )
    if b.shape != x.shape:
        raise ValueError(
            f"plot_llr_distribution(): llrs {x.shape} and bits {b.shape} differ "
            "in shape."
        )
    m = 1 if bits_per_symbol is None else int(bits_per_symbol)
    if m < 1 or (x.ndim and x.shape[-1] % m):
        raise ValueError(
            f"plot_llr_distribution(): bits_per_symbol must divide N = "
            f"{x.shape[-1] if x.ndim else 1}, got {bits_per_symbol}."
        )
    logger.debug("Generating LLR distribution plot (%s LLRs).", x.size)

    # Columns are the bit positions of the label; one column without
    # bits_per_symbol.
    L = x.reshape(-1, m)
    zero = b.reshape(-1, m) == 0
    one = b.reshape(-1, m) == 1
    lim = float(xp.percentile(xp.abs(L), 99.9)) or 1.0
    edges = xp.linspace(-lim, lim, bins + 1)

    def counts(mask):
        """Histogram per bit position, ``(m, bins)``, on the device."""
        return xp.stack(
            [xp.histogram(L[:, j][mask[:, j]], bins=edges)[0] for j in range(m)]
        )

    h0 = to_device(counts(zero), "cpu")
    h1 = to_device(counts(one), "cpu")
    n0 = to_device(xp.count_nonzero(zero, axis=0), "cpu")
    n1 = to_device(xp.count_nonzero(one, axis=0), "cpu")
    if n0.sum() + n1.sum() != L.size:
        raise ValueError("plot_llr_distribution(): bits must be 0 or 1.")
    if not n0.sum() or not n1.sum():
        raise ValueError("plot_llr_distribution(): bits must hold both 0 and 1.")
    edges_h = to_device(edges, "cpu")
    width = edges_h[1] - edges_h[0]
    centers = 0.5 * (edges_h[1:] + edges_h[:-1])

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=_grid_figsize(1, 2))
    else:
        axes = np.asarray(ax).flatten()[:2]
        fig = axes[0].figure
    ax_pdf, ax_cons = axes

    ax_pdf.stairs(h0.sum(axis=0) / (n0.sum() * width), edges_h, label="b = 0")
    ax_pdf.stairs(h1.sum(axis=0) / (n1.sum() * width), edges_h, label="b = 1")
    ax_pdf.set_xlabel("LLR L")
    ax_pdf.set_ylabel("Density p(L | b)")
    ax_pdf.set_title("Conditional LLR densities")
    ax_pdf.legend()

    span = [np.inf, -np.inf]  # measured LLR range, which the diagonal covers
    for j in range(m):
        if not n0[j] or not n1[j]:
            continue
        keep = (h0[j] >= _MIN_COUNT) & (h1[j] >= _MIN_COUNT)
        with np.errstate(divide="ignore"):
            ratio = np.log(h0[j][keep] / h1[j][keep]) + np.log(n1[j] / n0[j])
        label = "measured" if m == 1 else f"bit {j}"
        ax_cons.plot(centers[keep], ratio, "o-", markersize=3, label=label)
        if keep.any():
            span = [min(span[0], centers[keep][0]), max(span[1], centers[keep][-1])]
    if span[0] > span[1]:
        span = [-lim, lim]
    ax_cons.plot(span, span, "--", color="gray", label="consistent")
    ax_cons.set_xlabel("LLR L")
    ax_cons.set_ylabel("log p(L | b = 0) / p(L | b = 1)")
    ax_cons.set_title("Consistency")
    ax_cons.legend()

    if title is not None:
        fig.suptitle(title)
    return _finish((fig, axes), show)


def plot_parity_check(
    code: Any,
    *,
    ax: Any | None = None,
    title: str | None = None,
    show: bool = False,
) -> tuple[Any, Any] | None:
    """Plot the nonzero entries and the node degrees of a parity-check matrix.

    Panels: the matrix ``H``, one mark per nonzero entry (the circulant
    blocks of a quasi-cyclic code, the dual-diagonal parity part of DVB-S2,
    the punctured columns shaded), and the fractions of variable nodes
    (columns) and check nodes (rows) of each degree.

    Parameters
    ----------
    code : LDPCCode, LinearBlockCode or array_like
        A code with a parity-check matrix, or the matrix itself, ``(m, n)``
        with entries 0 and 1 (dense, or a ``scipy.sparse`` array).
    ax : array_like of matplotlib.axes.Axes, optional
        Two axes to draw on; ``None`` creates a figure.
    title : str, optional
        Title of the matrix panel; defaults to the code and the size of
        ``H``.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.

    Returns
    -------
    (fig, axes) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.

    Raises
    ------
    ValueError
        If the matrix is not 2-D or holds values other than 0 and 1.

    Notes
    -----
    Matrices of up to 10^6 entries are drawn as an image; larger ones as
    one rasterized marker per nonzero entry, so that a DVB-S2 matrix
    (32400 x 64800) is drawn from its 2 x 10^5 edges.
    """
    import scipy.sparse

    from ..coding import LDPCCode, LinearBlockCode

    punctured = 0
    if isinstance(code, LDPCCode):
        h = code.parity_check()
        punctured = code.punctured
        name = repr(code)
    elif isinstance(code, LinearBlockCode):
        h = scipy.sparse.csr_array(code.parity_check)
        name = repr(code)
    else:
        h = scipy.sparse.csr_array(code) if not scipy.sparse.issparse(code) else code
        name = "H"
    h = scipy.sparse.coo_array(h)
    if h.ndim != 2:
        raise ValueError(f"plot_parity_check(): H must be 2-D, got {h.ndim}-D.")
    h.sum_duplicates()
    h.eliminate_zeros()
    if h.nnz and not np.all(h.data == 1):
        raise ValueError("plot_parity_check(): H must hold only 0 and 1.")
    m, n = h.shape
    logger.debug("Generating parity-check plot (%s x %s, %s ones).", m, n, h.nnz)

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=_grid_figsize(1, 2))
    else:
        axes = np.asarray(ax).flatten()[:2]
        fig = axes[0].figure
    ax_h, ax_deg = axes

    if m * n <= _DENSE_ENTRIES:
        ax_h.spy(h.toarray(), aspect="auto")
    else:
        size = float(np.clip(50.0 / np.sqrt(max(m, n)), 0.05, 2.0))
        ax_h.spy(h, markersize=size, aspect="auto", rasterized=True)
    if punctured:
        ax_h.axvspan(n - punctured - 0.5, n - 0.5, color="gray", alpha=0.3)
        ax_h.text(n - punctured, -0.5, "punctured", va="bottom", fontsize=9)
    ax_h.tick_params(labeltop=False, labelbottom=True)
    ax_h.grid(False)
    ax_h.set_xlabel("Variable node (column)")
    ax_h.set_ylabel("Check node (row)")
    ax_h.set_title(title if title is not None else f"{name}: {m} x {n}, {h.nnz} ones")

    var_deg = np.bincount(h.col, minlength=n)
    chk_deg = np.bincount(h.row, minlength=m)
    for degrees, total, offset, label in (
        (var_deg, n, -0.2, "variable nodes"),
        (chk_deg, m, 0.2, "check nodes"),
    ):
        values, counts = np.unique(degrees, return_counts=True)
        ax_deg.bar(values + offset, counts / total, width=0.4, label=label)
    ax_deg.set_xlabel("Degree")
    ax_deg.set_ylabel("Fraction of nodes")
    ax_deg.set_title("Node degrees")
    ax_deg.legend()
    return _finish((fig, axes), show)


def plot_exit_chart(
    chart: Any,
    *,
    trajectory: bool = True,
    ax: Any | None = None,
    title: str | None = None,
    show: bool = False,
) -> tuple[Any, Any] | None:
    """Plot an EXIT chart: the two transfer curves and the decoding trajectory.

    The first curve ``I_E(I_A)`` and the second with its axes swapped, so
    that iterative decoding is a staircase between them: it converges when
    the tunnel between the curves stays open up to the top right, and
    otherwise stalls where they cross (marked).

    Parameters
    ----------
    chart : ExitChart
        The chart of :func:`commkit.coding.exit_chart`.
    trajectory : bool, default True
        Draw the decoding trajectory.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; ``None`` creates a figure.  Charts drawn on the same
        axes compare components or channel conditions.
    title : str, optional
        Title of the figure; defaults to whether the tunnel is open and the
        iterations of the trajectory.
    show : bool, default False
        Call ``plt.show()`` and return ``None``.

    Returns
    -------
    (fig, ax) or None
        The figure and its axes; ``None`` with ``show=True``, which
        shows the figure instead.

    Raises
    ------
    TypeError
        If ``chart`` is not an :class:`~commkit.coding.ExitChart`.
    """
    from ..coding import ExitChart

    if not isinstance(chart, ExitChart):
        raise TypeError(
            "plot_exit_chart(): chart must be an ExitChart, got "
            f"{type(chart).__name__}; build one with commkit.coding.exit_chart()."
        )
    logger.debug("Generating EXIT chart (converges: %s).", chart.converges)

    if ax is None:
        fig, axi = plt.subplots(figsize=_square_figsize())
    else:
        fig, axi = _get_axis(ax)

    first, second = chart.first, chart.second
    axi.plot(first.a_priori, first.extrinsic, label=first.label or "first")
    axi.plot(second.extrinsic, second.a_priori, label=second.label or "second")
    if trajectory:
        axi.plot(
            chart.trajectory[:, 0],
            chart.trajectory[:, 1],
            color="white",
            linewidth=0.8,
            alpha=0.8,
            label="trajectory",
        )
    if not chart.converges:
        x, y = chart.trajectory[-1]
        axi.plot([x], [y], "o", color="red", label="stalls")
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1)
    axi.set_aspect("equal")
    axi.set_xlabel("I_A (first) = I_E (second)")
    axi.set_ylabel("I_E (first) = I_A (second)")
    if title is None:
        state = "open" if chart.converges else "closed"
        title = f"EXIT chart: tunnel {state}, {chart.iterations} iterations"
    axi.set_title(title)
    axi.legend(loc="lower right", fontsize=8)
    return _finish((fig, axi), show)
