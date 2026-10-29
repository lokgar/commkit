"""Numba kernels of the coding module (private).

The per-codeword decoders and matchers that run on the CPU, parallel over
blocks: layered LDPC belief propagation (:func:`ldpc_layered`), polar
list decoding (:func:`polar_scl`), Viterbi and BCJR on a trellis
(:func:`viterbi`, :func:`bcjr`, :func:`bcjr_blocks`), turbo encoding and
decoding (:func:`turbo_encode`, :func:`turbo_decode`) and the CCDM
(:func:`ccdm_match`, :func:`ccdm_dematch`).  Imported on first use through
the ``_get_numba_*`` getters of :mod:`~commkit.coding.ldpc`,
:mod:`~commkit.coding.polar`, :mod:`~commkit.coding.convolutional`,
:mod:`~commkit.coding.turbo` and :mod:`~commkit.coding.shaping`, which
document each kernel, so ``import commkit`` does not load Numba.

The kernels sit at module level so that Numba's disk cache holds them: a
kernel built inside a function and calling another kernel built there
captures that dispatcher in its closure, which Numba's cache key cannot
reproduce, so it compiled again in every process.
"""

import numba
import numpy as np

from .shaping import _RANGE_BITS

# A step shared by a kernel, inlined into it before it is compiled.
_inline = numba.njit(cache=True, nogil=True, inline="always")


# -----------------------------------------------------------------------------
# Layered LDPC decoding
# -----------------------------------------------------------------------------

_PHI_MIN, _PHI_MAX = np.float32(1e-7), np.float32(30.0)
_HALF = np.float32(0.5)
_ZERO = np.float32(0.0)


@_inline
def phi(x):  # type: ignore[no-untyped-def]
    """``-log tanh(x / 2)``, ``x`` clipped to ``[1e-7, 30]``."""
    x = min(max(x, _PHI_MIN), _PHI_MAX)
    return -np.log(np.tanh(_HALF * x))


@_inline
def min_sum_row(a, r, cols, e0, e1, scale, offset):  # type: ignore[no-untyped-def]
    """Update one check row by min-sum (scaled, offset); ``a`` and ``r`` in place."""
    min1 = np.float32(np.inf)
    min2 = np.float32(np.inf)
    idx = -1
    neg = False
    for e in range(e0, e1):
        q = a[cols[e]] - r[e]
        neg ^= q < _ZERO
        mag = abs(q)
        if mag < min1:
            min2 = min1
            min1 = mag
            idx = e
        elif mag < min2:
            min2 = mag
    m1 = max(scale * min1 - offset, _ZERO)
    m2 = max(scale * min2 - offset, _ZERO)
    for e in range(e0, e1):
        c = cols[e]
        q = a[c] - r[e]
        mag = m2 if e == idx else m1
        rn = -mag if neg ^ (q < _ZERO) else mag
        a[c] = q + rn
        r[e] = rn


@_inline
def sum_product_row(a, r, cols, e0, e1):  # type: ignore[no-untyped-def]
    """Update one check row by sum-product (``phi`` form); ``a`` and ``r`` in place."""
    total = _ZERO
    neg = False
    for e in range(e0, e1):
        q = a[cols[e]] - r[e]
        neg ^= q < _ZERO
        total += phi(abs(q))
    for e in range(e0, e1):
        c = cols[e]
        q = a[c] - r[e]
        mag = phi(total - phi(abs(q)))
        rn = -mag if neg ^ (q < _ZERO) else mag
        a[c] = q + rn
        r[e] = rn


@_inline
def satisfied(a, row_ptr, cols):  # type: ignore[no-untyped-def]
    """Whether the hard decisions of ``a`` satisfy every check."""
    for row in range(row_ptr.shape[0] - 1):
        parity = False
        for e in range(row_ptr[row], row_ptr[row + 1]):
            parity ^= a[cols[e]] < _ZERO
        if parity:
            return False
    return True


@numba.njit(cache=True, nogil=True, parallel=True)
def ldpc_layered(  # type: ignore[no-untyped-def]
    app, row_ptr, cols, sum_product, scale, offset, max_iterations,
    early_stop, iters, valid,
):  # fmt: skip
    """Layered LDPC decoding of the rows of ``app`` in place (see ``_get_numba_ldpc``)."""
    m = row_ptr.shape[0] - 1
    for b in numba.prange(app.shape[0]):  # type: ignore[attr-defined,no-untyped-call]
        a = app[b]
        r = np.zeros(cols.shape[0], dtype=np.float32)
        it = 0
        ok = early_stop and satisfied(a, row_ptr, cols)
        while not ok and it < max_iterations:
            for row in range(m):
                if sum_product:
                    sum_product_row(a, r, cols, row_ptr[row], row_ptr[row + 1])
                else:
                    min_sum_row(
                        a,
                        r,
                        cols,
                        row_ptr[row],
                        row_ptr[row + 1],
                        scale,
                        offset,
                    )
            it += 1
            if early_stop:
                ok = satisfied(a, row_ptr, cols)
        if not early_stop:
            ok = satisfied(a, row_ptr, cols)
        iters[b] = it
        valid[b] = ok


# -----------------------------------------------------------------------------
# Polar list decoding
# -----------------------------------------------------------------------------


@_inline
def descend(llr, alpha, beta, i, levels):  # type: ignore[no-untyped-def]
    """SC LLRs from the root down to leaf ``i`` (``f`` for left, ``g`` for right children).

    Level ``s`` of ``alpha`` (length ``n``) is ``alpha[2^s : 2^(s+1)]``; the
    root level is the channel's ``llr``, shared by every path.
    """
    top = levels if i == 0 else 1
    if i:
        while not (i >> (top - 1)) & 1:
            top += 1
    for s in range(top, 0, -1):
        half = 1 << (s - 1)
        upper = llr if s == levels else alpha
        src = 0 if s == levels else 1 << s
        dst = half
        if (i >> (s - 1)) & 1:
            for j in range(half):
                a = upper[src + j]
                b = upper[src + half + j]
                alpha[dst + j] = b - a if beta[half + j] else b + a
        else:
            for j in range(half):
                a = upper[src + j]
                b = upper[src + half + j]
                m = min(abs(a), abs(b))
                alpha[dst + j] = -m if (a < 0) != (b < 0) else m


@_inline
def ascend(beta, scratch, i, bit, levels):  # type: ignore[no-untyped-def]
    """Propagate leaf ``i``'s decision ``bit`` up through the completed subtrees.

    Level ``s`` of ``beta`` (length ``n``) is ``beta[2^s : 2^(s+1)]``; the
    root's partial sums (after the last leaf) are never read.
    """
    scratch[0] = bit
    size = 1
    s = 0
    while s < levels and (i >> s) & 1:
        # Right child of size 2^s complete: parent = [left ^ right, right].
        for j in range(size):
            scratch[size + j] = scratch[j]
            scratch[j] ^= beta[size + j]
        size *= 2
        s += 1
    if size < beta.shape[0]:
        for j in range(size):
            beta[size + j] = scratch[j]


@numba.njit(cache=True, nogil=True, parallel=True)
def polar_scl(llrs, frozen, list_size, crc_g, crc_c, info, k_info, out_u, valid):  # type: ignore[no-untyped-def]
    """SC list decoding of the rows of ``llrs``, CRC-aided when ``crc_g`` is not empty."""
    rows, n = llrs.shape
    levels = 0
    while (1 << levels) < n:
        levels += 1
    r = crc_c.shape[0]
    for row in numba.prange(rows):  # type: ignore[attr-defined,no-untyped-call]
        cap = list_size
        llr = llrs[row]
        alpha = np.zeros((cap, n), dtype=np.float32)
        beta = np.zeros((cap, n), dtype=np.int8)
        u = np.zeros((cap, n), dtype=np.int8)
        pm = np.zeros(cap, dtype=np.float64)
        scratch = np.zeros(n, dtype=np.int8)
        # The candidates of a decision: path p keeps its hard decision
        # (candidate p) or takes the other bit (candidate p + alive).
        cand = np.empty(2 * cap, dtype=np.float64)
        hard = np.empty(cap, dtype=np.int8)
        order = np.empty(2 * cap, dtype=np.int64)
        chosen = np.empty(2 * cap, dtype=np.bool_)
        source = np.empty(cap, dtype=np.int64)  # the candidate in each slot
        moved = np.empty(cap, dtype=np.int64)
        buf_alpha = np.empty((cap, n), dtype=np.float32)
        buf_beta = np.empty((cap, n), dtype=np.int8)
        buf_u = np.empty((cap, n), dtype=np.int8)
        alive = 1
        for i in range(n):
            for p in range(alive):
                descend(llr, alpha[p], beta[p], i, levels)
            if frozen[i]:
                for p in range(alive):
                    l0 = alpha[p, 1]
                    if l0 < 0:
                        pm[p] += -l0
                    u[p, i] = 0
                    ascend(beta[p], scratch, i, 0, levels)
                continue
            total = 2 * alive
            for p in range(alive):
                l0 = alpha[p, 1]
                hard[p] = 1 if l0 < 0 else 0
                cand[p] = pm[p]
                cand[p + alive] = pm[p] + abs(l0)
            # Stable sort by metric (insertion: at most 2 cap candidates).
            for a in range(total):
                v = cand[a]
                b = a
                while b > 0 and cand[order[b - 1]] > v:
                    order[b] = order[b - 1]
                    b -= 1
                order[b] = a
            keep = min(cap, total)
            chosen[:total] = False
            for a in range(keep):
                chosen[order[a]] = True
            # Survivors below keep stay in their slot; the others fill the
            # free slots in order.
            for d in range(keep):
                source[d] = d
            d = 0
            for a in range(keep, total):
                if not chosen[a]:
                    continue
                while chosen[d]:
                    d += 1
                source[d] = a
                chosen[d] = True
            # Only a path that changes slot is copied (on a fork, or into a
            # free slot), through the buffer, so that no path is overwritten
            # before every slot it feeds has read it.
            num = 0
            for d in range(keep):
                par = source[d] % alive
                if par != d:
                    buf_alpha[num] = alpha[par]
                    buf_beta[num] = beta[par]
                    buf_u[num, :i] = u[par, :i]
                    moved[num] = d
                    num += 1
            for a in range(num):
                d = moved[a]
                alpha[d] = buf_alpha[a]
                beta[d] = buf_beta[a]
                u[d, :i] = buf_u[a, :i]
            for d in range(keep):
                a = source[d]
                pm[d] = cand[a]
                u[d, i] = hard[a] if a < alive else 1 - hard[a - alive]
            alive = keep
            for p in range(alive):
                ascend(beta[p], scratch, i, u[p, i], levels)
        # Final choice: best metric, among CRC passes when there is a CRC.
        best = -1
        ok = False
        for p in range(alive):
            passes = True
            if r:
                for b in range(r):
                    acc = crc_c[b]
                    for t in range(k_info):
                        acc ^= u[p, info[t]] & crc_g[t, b]
                    if acc != u[p, info[k_info + b]]:
                        passes = False
                        break
            if passes and (best < 0 or pm[p] < pm[best]):
                best = p
                ok = True
        if best < 0:
            best = 0
            for p in range(1, alive):
                if pm[p] < pm[best]:
                    best = p
        for j in range(n):
            out_u[row, j] = u[best, j]
        valid[row] = ok


# -----------------------------------------------------------------------------
# Trellis decoding: Viterbi and BCJR
# -----------------------------------------------------------------------------

_NEG = -1.0e300


@numba.njit(cache=True, nogil=True)
def branch(llr_t, outputs, s, u):  # type: ignore[no-untyped-def]
    """Correlation of a trellis branch's outputs with the step's channel LLRs."""
    acc = 0.0
    for j in range(outputs.shape[2]):
        v = llr_t[j]
        acc += -v if outputs[s, u, j] else v
    return acc


@numba.njit(cache=True, nogil=True)
def combine(a, b, exact):  # type: ignore[no-untyped-def]
    """``log(e^a + e^b)`` (exact) or ``max(a, b)``."""
    if a < b:
        a, b = b, a
    if b <= _NEG or not exact:
        return a
    return a + np.log1p(np.exp(b - a))


@numba.njit(cache=True, nogil=True)
def viterbi_one(llr, nxt, outputs, start, end, decisions, record, metric, nm):  # type: ignore[no-untyped-def]
    """Viterbi pass from ``start`` to ``end``: the path metric at ``end``.

    With ``record``, ``decisions[t, q]`` holds ``2 s + u`` of the survivor
    into state ``q`` at step ``t + 1`` (written where ``q`` is reachable).
    """
    steps = llr.shape[0]
    states = nxt.shape[0]
    for s in range(states):
        metric[s] = 0.0 if s == start else _NEG
    for t in range(steps):
        for s in range(states):
            nm[s] = _NEG
        for s in range(states):
            if metric[s] <= _NEG:
                continue
            for u in range(2):
                q = nxt[s, u]
                v = metric[s] + branch(llr[t], outputs, s, u)
                if v > nm[q]:
                    nm[q] = v
                    if record:
                        decisions[t, q] = s * 2 + u
        for s in range(states):
            metric[s] = nm[s]
    return metric[end]


# Tail-biting decoding scores all start states in one pass up to this many
# states (two float64 (states, states) matrices per block: 16 MB at 1024).
_JOINT_STATES = 1024


@numba.njit(cache=True, nogil=True)
def tail_biting_start(llr, nxt, outputs, cur, new):  # type: ignore[no-untyped-def]
    """The best start state of a tail-biting block (``-1`` if none ends where it starts).

    The Viterbi passes of every start in one: ``cur[q, st]`` is the metric
    of state ``q`` from start ``st``, each branch metric computed once per
    step.  The same sums in the same order as one pass per start.
    """
    steps = llr.shape[0]
    states = nxt.shape[0]
    for q in range(states):
        for st in range(states):
            cur[q, st] = 0.0 if q == st else _NEG
    for t in range(steps):
        new[:, :] = _NEG
        for s in range(states):
            for u in range(2):
                q = nxt[s, u]
                b = branch(llr[t], outputs, s, u)
                for st in range(states):
                    c = cur[s, st]
                    if c > _NEG:
                        v = c + b
                        if v > new[q, st]:
                            new[q, st] = v
        cur, new = new, cur
    best = -1
    best_val = _NEG
    for st in range(states):
        if cur[st, st] > best_val:
            best_val = cur[st, st]
            best = st
    return best


@numba.njit(cache=True, nogil=True, parallel=True)
def viterbi(llrs, nxt, outputs, tail_biting, info_steps, out_bits):  # type: ignore[no-untyped-def]
    """Viterbi decoding of the rows of ``llrs`` (see ``_get_numba_trellis``).

    Tail-biting decoding scores every start state (ending in the same
    state) without decisions, then records them for the best start only.
    """
    rows, steps = llrs.shape[0], llrs.shape[1]
    states = nxt.shape[0]
    joint = tail_biting and states <= _JOINT_STATES
    size = states if joint else 0
    for row in numba.prange(rows):  # type: ignore[attr-defined,no-untyped-call]
        # 2 s + u < 2^16 for the 2^15 states of K = 16.
        decisions = np.zeros((steps, states), dtype=np.uint16)
        metric = np.empty(states)
        nm = np.empty(states)
        best = -1
        if tail_biting:
            if joint:
                best = tail_biting_start(
                    llrs[row],
                    nxt,
                    outputs,
                    np.empty((size, size)),
                    np.empty((size, size)),
                )
            else:
                best_val = _NEG
                for start in range(states):
                    val = viterbi_one(
                        llrs[row],
                        nxt,
                        outputs,
                        start,
                        start,
                        decisions,
                        False,
                        metric,
                        nm,
                    )
                    if val > best_val:
                        best_val = val
                        best = start
            if best >= 0:
                viterbi_one(
                    llrs[row], nxt, outputs, best, best, decisions, True, metric, nm
                )
        elif (
            viterbi_one(llrs[row], nxt, outputs, 0, 0, decisions, True, metric, nm)
            > _NEG
        ):
            best = 0
        if best < 0:
            decisions[:, :] = 0  # no path reaches an end state
        s = max(best, 0)
        for t in range(steps - 1, -1, -1):
            code = np.int64(decisions[t, s])
            if t < info_steps:
                out_bits[row, t] = code & 1
            s = code >> 1


@numba.njit(cache=True, nogil=True)
def bcjr(llr, apriori, nxt, outputs, tail, forced, start, end, exact, app_u, app_c):  # type: ignore[no-untyped-def]
    """BCJR over one block: a-posteriori LLRs of the inputs and the outputs."""
    steps = llr.shape[0]
    states = nxt.shape[0]
    r = outputs.shape[2]
    alpha = np.full((steps + 1, states), _NEG)
    beta = np.full((steps + 1, states), _NEG)
    alpha[0, start] = 0.0
    beta[steps, end] = 0.0
    gamma = np.full((steps, states, 2), _NEG)
    for t in range(steps):
        for s in range(states):
            for u in range(2):
                if forced[t] and u != tail[s]:
                    continue
                g = 0.5 * branch(llr[t], outputs, s, u)
                g += -0.5 * apriori[t] if u else 0.5 * apriori[t]
                gamma[t, s, u] = g

    for t in range(steps):
        for s in range(states):
            if alpha[t, s] <= _NEG:
                continue
            for u in range(2):
                if gamma[t, s, u] <= _NEG:
                    continue
                q = nxt[s, u]
                alpha[t + 1, q] = combine(
                    alpha[t + 1, q], alpha[t, s] + gamma[t, s, u], exact
                )
    for t in range(steps - 1, -1, -1):
        for s in range(states):
            acc = _NEG
            for u in range(2):
                if gamma[t, s, u] <= _NEG:
                    continue
                acc = combine(acc, gamma[t, s, u] + beta[t + 1, nxt[s, u]], exact)
            beta[t, s] = acc
    for t in range(steps):
        zu = _NEG
        ou = _NEG
        for j in range(r):
            app_c[t, j] = 0.0
        c0 = np.full(r, _NEG)
        c1 = np.full(r, _NEG)
        for s in range(states):
            for u in range(2):
                if gamma[t, s, u] <= _NEG or alpha[t, s] <= _NEG:
                    continue
                v = alpha[t, s] + gamma[t, s, u] + beta[t + 1, nxt[s, u]]
                if u:
                    ou = combine(ou, v, exact)
                else:
                    zu = combine(zu, v, exact)
                for j in range(r):
                    if outputs[s, u, j]:
                        c1[j] = combine(c1[j], v, exact)
                    else:
                        c0[j] = combine(c0[j], v, exact)
        app_u[t] = zu - ou
        for j in range(r):
            app_c[t, j] = c0[j] - c1[j]


@numba.njit(cache=True, nogil=True, parallel=True)
def bcjr_blocks(llrs, nxt, outputs, tail, forced, start, end, exact, app_u, app_c):  # type: ignore[no-untyped-def]
    """BCJR over the rows of ``llrs``, without a-priori information."""
    apriori = np.zeros(llrs.shape[1])
    for row in numba.prange(llrs.shape[0]):  # type: ignore[attr-defined,no-untyped-call]
        bcjr(llrs[row], apriori, nxt, outputs, tail, forced, start, end, exact,
             app_u[row], app_c[row])  # fmt: skip


# -----------------------------------------------------------------------------
# Turbo codes
# -----------------------------------------------------------------------------


@numba.njit(cache=True, nogil=True)
def rsc(bits, nxt, outputs, tail, parity, tail_bits):  # type: ignore[no-untyped-def]
    """Parity of ``bits`` and the 3 tail (systematic, parity) pairs."""
    s = 0
    for t in range(bits.shape[0]):
        u = bits[t]
        parity[t] = outputs[s, u, 1]
        s = nxt[s, u]
    for t in range(3):
        u = tail[s]
        tail_bits[t, 0] = u
        tail_bits[t, 1] = outputs[s, u, 1]
        s = nxt[s, u]


@numba.njit(cache=True, nogil=True, parallel=True)
def turbo_encode(bits, pi, nxt, outputs, tail, out):  # type: ignore[no-untyped-def]
    """LTE turbo encoding of the rows of ``bits`` (see ``_get_numba_turbo``)."""
    k = bits.shape[1]
    for row in numba.prange(bits.shape[0]):  # type: ignore[attr-defined,no-untyped-call]
        p1 = np.empty(k, dtype=np.int8)
        p2 = np.empty(k, dtype=np.int8)
        t1 = np.empty((3, 2), dtype=np.int8)
        t2 = np.empty((3, 2), dtype=np.int8)
        permuted = np.empty(k, dtype=np.int8)
        for i in range(k):
            permuted[i] = bits[row, pi[i]]
        rsc(bits[row], nxt, outputs, tail, p1, t1)
        rsc(permuted, nxt, outputs, tail, p2, t2)
        d0 = out[row, : k + 4]
        d1 = out[row, k + 4 : 2 * k + 8]
        d2 = out[row, 2 * k + 8 :]
        d0[:k] = bits[row]
        d1[:k] = p1
        d2[:k] = p2
        # TS 36.212 clause 5.1.3.2.2.
        d0[k], d1[k], d2[k] = t1[0, 0], t1[0, 1], t1[1, 0]
        d0[k + 1], d1[k + 1], d2[k + 1] = t1[1, 1], t1[2, 0], t1[2, 1]
        d0[k + 2], d1[k + 2], d2[k + 2] = t2[0, 0], t2[0, 1], t2[1, 0]
        d0[k + 3], d1[k + 3], d2[k + 3] = t2[1, 1], t2[2, 0], t2[2, 1]


@numba.njit(cache=True, nogil=True, parallel=True)
def turbo_decode(llrs, pi, nxt, outputs, tail, iterations, exact, scale, out):  # type: ignore[no-untyped-def]
    """Iterative turbo decoding of the rows of ``llrs`` (see ``_get_numba_turbo``)."""
    k = pi.shape[0]
    steps = k + 3
    forced = np.zeros(steps, dtype=np.bool_)
    forced[k:] = True
    for row in numba.prange(llrs.shape[0]):  # type: ignore[attr-defined,no-untyped-call]
        d0 = llrs[row, : k + 4]
        d1 = llrs[row, k + 4 : 2 * k + 8]
        d2 = llrs[row, 2 * k + 8 :]
        ch1 = np.empty((steps, 2))
        ch2 = np.empty((steps, 2))
        for t in range(k):
            ch1[t, 0] = d0[t]
            ch1[t, 1] = d1[t]
            ch2[t, 0] = d0[pi[t]]
            ch2[t, 1] = d2[t]
        ch1[k, 0], ch1[k, 1], ch1[k + 1, 0] = d0[k], d1[k], d2[k]
        ch1[k + 1, 1], ch1[k + 2, 0], ch1[k + 2, 1] = (
            d0[k + 1],
            d1[k + 1],
            d2[k + 1],
        )
        ch2[k, 0], ch2[k, 1], ch2[k + 1, 0] = d0[k + 2], d1[k + 2], d2[k + 2]
        ch2[k + 1, 1], ch2[k + 2, 0], ch2[k + 2, 1] = (
            d0[k + 3],
            d1[k + 3],
            d2[k + 3],
        )
        la1 = np.zeros(steps)
        la2 = np.zeros(steps)
        app1 = np.empty(steps)
        app2 = np.empty(steps)
        app_c = np.empty((steps, 2))
        for _ in range(iterations):
            bcjr(ch1, la1, nxt, outputs, tail, forced, 0, 0, exact, app1, app_c)
            for t in range(k):
                # Extrinsic of decoder 1, kept in la1 (natural order).
                la1[t] = scale * (app1[t] - la1[t] - ch1[t, 0])
            for t in range(k):
                la2[t] = la1[pi[t]]
            bcjr(ch2, la2, nxt, outputs, tail, forced, 0, 0, exact, app2, app_c)
            for t in range(k):
                la1[pi[t]] = scale * (app2[t] - la2[t] - ch2[t, 0])
        for t in range(k):
            out[row, pi[t]] = 1 if app2[t] < 0 else 0


# -----------------------------------------------------------------------------
# Constant-composition distribution matching
# -----------------------------------------------------------------------------

_TOP = np.int64(1) << _RANGE_BITS
_HALF_RANGE = np.int64(1) << (_RANGE_BITS - 1)


@numba.njit(cache=True, nogil=True, inline="always")
def largest(counts):  # type: ignore[no-untyped-def]
    """Index of the largest count (the first on ties)."""
    big = 0
    for i in range(1, counts.shape[0]):
        if counts[i] > counts[big]:
            big = i
    return big


@numba.njit(cache=True, nogil=True, parallel=True)
def ccdm_match(bits, composition, out):  # type: ignore[no-untyped-def]
    """Match the rows of ``bits`` to point indices of the composition (range coder)."""
    k = bits.shape[1]
    n = out.shape[1]
    for b in numba.prange(bits.shape[0]):  # type: ignore[attr-defined,no-untyped-call]
        counts = composition.copy()
        rng = _TOP
        d = np.int64(0)
        pos = 0
        for _ in range(_RANGE_BITS):
            d = (d << 1) | (bits[b, pos] if pos < k else 0)
            pos += 1
        remaining = np.int64(n)
        for t in range(n):
            big = largest(counts)
            rest = rng
            for i in range(counts.shape[0]):
                rest -= rng * counts[i] // remaining
            cum = np.int64(0)
            s = 0
            share = np.int64(0)
            for i in range(counts.shape[0]):
                share = rng * counts[i] // remaining
                if i == big:
                    share += rest
                if d < cum + share:
                    s = i
                    break
                cum += share
            out[b, t] = s
            d -= cum
            rng = share
            counts[s] -= 1
            remaining -= 1
            while rng < _HALF_RANGE:
                rng <<= 1
                d = (d << 1) | (bits[b, pos] if pos < k else 0)
                pos += 1


@numba.njit(cache=True, nogil=True, parallel=True)
def ccdm_dematch(symbols, composition, buffer_bits, bits, valid):  # type: ignore[no-untyped-def]
    """Dematch the rows of ``symbols`` back to bits, flagging invalid rows."""
    k = bits.shape[1]
    n = symbols.shape[1]
    for b in numba.prange(symbols.shape[0]):  # type: ignore[attr-defined,no-untyped-call]
        counts = composition.copy()
        low = np.zeros(buffer_bits + 1, dtype=np.uint8)  # low[j]: bit 2^-j
        rng = _TOP
        shift = _RANGE_BITS
        remaining = np.int64(n)
        ok = True
        for t in range(n):
            s = symbols[b, t]
            if s < 0 or s >= counts.shape[0] or counts[s] == 0:
                ok = False
                break
            big = largest(counts)
            rest = rng
            for i in range(counts.shape[0]):
                rest -= rng * counts[i] // remaining
            cum = np.int64(0)
            for i in range(s):
                cum += rng * counts[i] // remaining
                if i == big:
                    cum += rest
            share = rng * counts[s] // remaining
            if s == big:
                share += rest
            # low += cum 2^-shift, with carries into the higher bits.
            carry = 0
            j = shift
            while (cum > 0 or carry) and j > 0:
                v = low[j] + (cum & 1) + carry
                low[j] = v & 1
                carry = v >> 1
                cum >>= 1
                j -= 1
            rng = share
            counts[s] -= 1
            remaining -= 1
            while rng < _HALF_RANGE:
                rng <<= 1
                shift += 1
            if shift > buffer_bits:
                ok = False
                break
        # v = ceil(low 2^k): the first k bits, plus one if more follow.
        beyond = 0
        for j in range(k + 1, buffer_bits + 1):
            if low[j]:
                beyond = 1
                break
        up = beyond
        for j in range(k, 0, -1):
            v = low[j] + up
            bits[b, j - 1] = v & 1
            up = v >> 1
        # A sequence of the composition is the image of a payload only if v
        # 2^-k lies in the final interval [low, low + rng 2^-shift): the
        # composition has more sequences than the 2^k payloads.  The carry
        # out of the k bits means v = 2^k, no payload.
        if ok and up:
            ok = False
        if ok and beyond:
            # d = (v 2^-k - low) 2^shift = 2^(shift - k) - F, with F the bits
            # of low after the k-th.  d < rng <= 2^32 needs every bit of F
            # above its lowest 32 to be 1, and then d = 2^w - (those 32 bits).
            lo = max(k + 1, shift - _RANGE_BITS + 1)
            for j in range(k + 1, lo):
                if not low[j]:
                    ok = False
                    break
            if ok:
                f = np.int64(0)
                for j in range(lo, shift + 1):
                    f = (f << 1) | low[j]
                ok = (np.int64(1) << (shift - lo + 1)) - f < rng
        valid[b] = ok
