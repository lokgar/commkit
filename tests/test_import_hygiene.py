"""Process-level checks: importing and using commkit has no global side effects.

Each check runs in a fresh interpreter, because the test process itself has
already imported Matplotlib, CuPy and commkit.
"""

import subprocess
import sys

import pytest


def _run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    return out.stdout.strip().splitlines()[-1]


def test_import_configures_nothing_global():
    """No log handlers and no warning filters of commkit's own.

    NumPy and SciPy install a few warning filters when imported; they are
    imported first so that only commkit's effect is measured.
    """
    result = _run(
        "import logging, warnings\n"
        "import numpy, scipy.signal, scipy.special, scipy.ndimage\n"
        "before = list(warnings.filters)\n"
        "import commkit\n"
        "print(logging.getLogger('commkit').handlers == [],"
        " logging.getLogger().handlers == [],"
        " warnings.filters == before)"
    )
    assert result == "True True True"


def test_import_does_not_load_matplotlib_or_cupy():
    """``import commkit`` loads neither Matplotlib nor CuPy (no GPU access)."""
    result = _run(
        "import sys, commkit\nprint('matplotlib' in sys.modules, 'cupy' in sys.modules)"
    )
    assert result == "False False"


def test_import_leaves_matplotlib_defaults_unchanged():
    """Even with plotting loaded, the commkit theme is applied only on request."""
    result = _run(
        "import matplotlib as mpl\n"
        "before = dict(mpl.rcParams)\n"
        "import commkit.plotting\n"
        "print(dict(mpl.rcParams) == before)"
    )
    assert result == "True"


@pytest.mark.parametrize(
    "module",
    [
        "commkit.analysis",
        "commkit.equalization",
        "commkit.filtering",
        "commkit.frequency",
        "commkit.impairments",
        "commkit.mapping",
        "commkit.metrics",
        "commkit.multirate",
        "commkit.recovery",
        "commkit.spectral",
        "commkit.timing",
    ],
)
def test_numerical_modules_do_not_import_matplotlib(module):
    result = _run(f"import sys, {module}\nprint('matplotlib' in sys.modules)")
    assert result == "False"


def test_host_signal_never_touches_cupy():
    """Constructing and processing a NumPy Signal never imports CuPy."""
    result = _run(
        "import sys, numpy as np, commkit\n"
        "from commkit.impairments import apply_awgn\n"
        "sig = commkit.Signal(samples=np.ones(64, np.complex64),"
        " sampling_rate=2.0, symbol_rate=1.0)\n"
        "apply_awgn(sig, esn0_db=20, rng=1)\n"
        "print('cupy' in sys.modules)"
    )
    assert result == "False"


def test_subpackages_load_on_attribute_access():
    result = _run(
        "import sys, commkit\n"
        "loaded = 'commkit.plotting' in sys.modules\n"
        "commkit.plotting.plot_psd\n"
        "print(loaded, 'commkit.plotting' in sys.modules)"
    )
    assert result == "False True"


@pytest.mark.gpu_only
def test_gpu_dispatch_emits_no_cupy_jit_warning(backend_device):
    """CuPy's experimental-JIT FutureWarning (raised when cupyx.scipy.signal is
    imported) must not reach users on their first GPU call."""
    result = _run(
        "import warnings\n"
        "warnings.simplefilter('error', FutureWarning)\n"
        "import cupy as cp\n"
        "from commkit.backend import dispatch\n"
        "dispatch(cp.zeros(4))\n"
        "print('ok')"
    )
    assert result == "ok"
