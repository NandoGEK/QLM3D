"""Stochastic series expansion: a Markov chain of operator strings (Yan et al., PRB 99, 165135)."""

import os
import weakref
from typing import Any

import llvmlite.ir as lir
import numpy as np
from numba import _helperlib, njit
from numba.core import cgutils, types
from numba.extending import intrinsic

from lattice import SIGNS

IDENTITY, DIAG, FLIP = 0, 1, 2  # the types of operator in a slot
EMPTY: tuple[int, int] = (IDENTITY, -1)  # a slot without operator

NO_NUMBERS = np.zeros(0)  # no injected random numbers: the generator is used
REC_FLIP = np.uint32(1 << 31)
REC_FP = np.uint32(1 << 30)
REC_NEXT = np.uint32((1 << 30) - 1)
U32_FFFF = np.uint32(0xFFFF)
U32_16 = np.uint32(16)
U64_1 = np.uint64(1)
U64_ALL = np.uint64(0xFFFFFFFFFFFFFFFF)

# ---------------------------------------------------------------- random numbers

_generator_owner: Any = None  # weak reference to the chain whose stream numba's generator holds


def use_generator_of(sse):
    """Make numba's generator continue the stream of `sse` (chains in one process take turns)."""
    global _generator_owner
    owner = _generator_owner() if _generator_owner is not None else None
    if owner is sse:
        return
    pointer = _helperlib.rnd_get_np_state_ptr()  # pyright: ignore[reportAttributeAccessIssue]
    if owner is not None:
        owner.generator_state = _helperlib.rnd_get_state(pointer)  # pyright: ignore[reportAttributeAccessIssue]
    if sse.generator_state is not None:
        _helperlib.rnd_set_state(pointer, sse.generator_state)  # pyright: ignore[reportAttributeAccessIssue]
    _generator_owner = weakref.ref(sse)


@njit(cache=True)
def seed_generator(seed):
    """Seed numba's generator."""
    np.random.seed(seed)


@njit(cache=True, inline="always")
def uniform(numbers, position):
    """The next random number in [0, 1): from the generator, or from `numbers` if it is not empty."""
    if numbers.shape[0] > 0:
        i = position[0]
        position[0] = i + 1
        return numbers[i]
    return np.random.random()


@njit(cache=True, inline="always")
def random_below(n, numbers, position):
    """A random integer in [0, n), as int(u n) from one random number u."""
    value = int(uniform(numbers, position) * n)
    if value >= n:
        value = n - 1
    return value


# ---------------------------------------------------------------- the string


@njit(cache=True)
def flip_links(s, links, p):
    """Flip plaquette p of s in place."""
    for k in range(4):
        s[links[p, k]] = -s[links[p, k]]


@njit(cache=True)
def orientation(s, links, p):
    """+1 if U_p can act on plaquette p of s, -1 if U_p^dag can, 0 if p is not flippable."""
    raisable = True
    lowerable = True
    for k in range(4):
        v = s[links[p, k]]
        sign = 1 if k < 2 else -1
        raisable &= v == -sign
        lowerable &= v == sign
    return 1 if raisable else (-1 if lowerable else 0)


@njit(cache=True)
def string_is_valid(state, ops_t, ops_p, links):
    """Whether every FLIP acts on a flippable plaquette and the last state equals the first."""
    s = state.copy()
    for x in range(ops_t.shape[0]):
        if ops_t[x] == FLIP:
            if orientation(s, links, ops_p[x]) == 0:
                return False
            flip_links(s, links, ops_p[x])
    return bool(np.all(s == state))


@njit(cache=True)
def string_k_vector(state, ops_t, ops_p, links, n_plaq):
    """Per plaquette the number of U_p minus the number of U_p^dag in the string."""
    k = np.zeros(n_plaq, np.int64)
    s = state.copy()
    for x in range(ops_t.shape[0]):
        if ops_t[x] == FLIP:
            p = ops_p[x]
            k[p] += orientation(s, links, p)
            flip_links(s, links, p)
    return k


@njit(cache=True)
def string_operator_states(state, ops_t, ops_p, links):
    """The state entering each operator of the string, one row per operator."""
    n = 0
    for x in range(ops_t.shape[0]):
        if ops_t[x] != IDENTITY:
            n += 1
    out = np.empty((n, state.shape[0]), np.int8)
    s = state.copy()
    j = 0
    for x in range(ops_t.shape[0]):
        if ops_t[x] == IDENTITY:
            continue
        out[j] = s
        j += 1
        if ops_t[x] == FLIP:
            flip_links(s, links, ops_p[x])
    return out


@njit(cache=True)
def string_mode(state, ops_t, ops_p, links, w):
    """sum_l w_l s_l on the state entering each operator of the string."""
    n = 0
    for x in range(ops_t.shape[0]):
        if ops_t[x] != IDENTITY:
            n += 1
    out = np.empty(n, np.complex128)
    s = state.copy()
    value = 0j
    for link in range(s.shape[0]):
        value += w[link] * s[link]
    j = 0
    for x in range(ops_t.shape[0]):
        if ops_t[x] == IDENTITY:
            continue
        out[j] = value
        j += 1
        if ops_t[x] == FLIP:
            p = ops_p[x]
            for k in range(4):
                link = links[p, k]
                value -= 2.0 * w[link] * s[link]
                s[link] = -s[link]
    return out


@njit(cache=True)
def rotate_string(state, ops_t, ops_p, links, numbers, position):
    """Rotate the string by a random number k of slots: `state` becomes the state entering slot k."""
    k = random_below(ops_t.shape[0], numbers, position)
    for x in range(k):
        if ops_t[x] == FLIP:
            flip_links(state, links, ops_p[x])
    rotated_t = np.concatenate((ops_t[k:], ops_t[:k]))
    rotated_p = np.concatenate((ops_p[k:], ops_p[:k]))
    ops_t[:] = rotated_t
    ops_p[:] = rotated_p


# ---------------------------------------------------------------- the index


def _prefetch_row_impl(typingctx, arr, idx):
    """prefetch_row(arr, idx): ask the CPU to load row idx of the 2-D array arr, which changes no result."""
    # The records a walk reads are far apart in memory and each is known only after the one before,
    # so the walk asks for its most likely next record while it works on the current one.
    if not isinstance(arr, types.Array) or arr.ndim != 2:
        return None
    sig = types.void(arr, types.intp)

    def codegen(context, builder, signature, args):
        """Emit the prefetch instruction for the address of row idx."""
        array = context.make_array(signature.args[0])(context, builder, args[0])
        strides = cgutils.unpack_tuple(builder, array.strides, 2)
        ptr = cgutils.pointer_add(builder, array.data, builder.mul(args[1], strides[0]))
        i8p = lir.IntType(8).as_pointer()
        i32 = lir.IntType(32)
        fnty = lir.FunctionType(lir.VoidType(), [i8p, i32, i32, i32])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.prefetch.p0")
        # read, high temporal locality, data cache
        builder.call(
            fn,
            [
                builder.bitcast(ptr, i8p),
                lir.Constant(i32, 0),
                lir.Constant(i32, 3),
                lir.Constant(i32, 1),
            ],
        )
        return context.get_dummy_value()

    return sig, codegen


prefetch_row: Any = intrinsic(_prefetch_row_impl)


@njit(cache=True, inline="always")
def flippable4(s, links4, q):
    """Whether plaquette q of s is flippable, where links4[4 q + k] is its k-th link."""
    raisable = True
    lowerable = True
    b = 4 * q
    for k in range(4):
        v = s[links4[b + k]]
        sign = 1 if k < 2 else -1
        raisable &= v == -sign
        lowerable &= v == sign
    return raisable or lowerable


@njit(cache=True, inline="always")
def write_record(j, x, q, flip, fp, links4, rec, slot_of, link_count, first_op, parity, fp_mask):
    """Write the record of operator j (slot x, plaquette q) except its next-operator pointers."""
    # Returns False if a link has more operators than `parity` has bits.
    ok = True
    words = parity.shape[1]
    r0 = np.uint32(0)
    if flip:
        r0 |= REC_FLIP
    if fp:
        r0 |= REC_FP
        fp_mask[j >> 6] |= U64_1 << np.uint64(j & 63)
    rec[j, 0] = r0
    slot_of[j] = np.uint32(x)
    b = 4 * q
    for k in range(4):
        link = np.int64(links4[b + k])
        c = link_count[link]
        if c == 0:
            first_op[link] = np.uint32(j)
        # bit c + 1 = parity of the FLIPs among the first c + 1 operators = bit c xor flip
        c1 = c + 1
        w1 = c1 >> 6
        if w1 < words:
            bit = (parity[link, c >> 6] >> np.uint64(c & 63)) & U64_1
            if flip:
                bit ^= U64_1
            parity[link, w1] |= bit << np.uint64(c1 & 63)
        else:
            ok = False
        if k & 1:
            rec[j, 4 + (k >> 1)] |= np.uint32(c) << U32_16
            rec[j, 6 + (k >> 1)] |= np.uint32(link) << U32_16
        else:
            rec[j, 4 + (k >> 1)] = np.uint32(c)
            rec[j, 6 + (k >> 1)] = np.uint32(link)
        link_count[link] = c1
    return ok


@njit(cache=True, nogil=True)
def diagonal_update_and_index(
    state, ops_t, ops_p, n, beta, n_plaq, links4, w_flippable, w_blocked, numbers, position,
    update, rec, slot_of, link_count, first_op, parity, fenwick, fp_mask, last_seen,
):  # fmt: skip
    """Go through the string once: insert and remove DIAGs (if `update`) and build the index."""
    # Returns (n, number of flippable vertices, largest number of operators on a link, ok). ok is
    # False if a link has more operators than `parity` has bits. The caller then enlarges `parity`
    # and calls again with update = False.
    M = ops_t.shape[0]
    n_links = link_count.shape[0]
    for link in range(n_links):
        link_count[link] = 0
        first_op[link] = np.uint32(M + 1)
        for w in range(parity.shape[1]):
            parity[link, w] = np.uint64(0)
    for block in range(fp_mask.shape[0]):
        fp_mask[block] = np.uint64(0)
    s = state.copy()
    j = 0
    n_fp = 0
    ok = True
    for x in range(M):
        t = ops_t[x]
        if t == IDENTITY:
            if update:  # try to insert a DIAG on a random plaquette
                q = random_below(n_plaq, numbers, position)
                fp = flippable4(s, links4, q)
                weight = w_flippable if fp else w_blocked
                if uniform(numbers, position) < min(1.0, beta * n_plaq * weight / (M - n)):
                    ops_t[x] = DIAG
                    ops_p[x] = q
                    n += 1
                    ok &= write_record(
                        j, x, q, False, fp, links4, rec, slot_of, link_count, first_op, parity,
                        fp_mask,
                    )  # fmt: skip
                    n_fp += 1 if fp else 0
                    j += 1
            continue
        q = ops_p[x]
        if t == DIAG:
            fp = flippable4(s, links4, q)
            if update:  # try to remove it
                weight = w_flippable if fp else w_blocked
                if uniform(numbers, position) < min(1.0, (M - n + 1) / (beta * n_plaq * weight)):
                    ops_t[x] = IDENTITY
                    ops_p[x] = -1
                    n -= 1
                    continue
            flip = False
        else:
            fp = True
            flip = True
            b = 4 * q
            for k in range(4):
                link = links4[b + k]
                s[link] = -s[link]
        ok &= write_record(
            j, x, q, flip, fp, links4, rec, slot_of, link_count, first_op, parity, fp_mask
        )
        n_fp += 1 if fp else 0
        j += 1
    n_ops = j
    most = 0
    for link in range(n_links):
        most = max(most, link_count[link])
        last_seen[link] = np.uint32(n_ops + 1)
    # backwards: the next operator on each link of every operator
    for j in range(n_ops - 1, -1, -1):
        lw0 = rec[j, 6]
        lw1 = rec[j, 7]
        l0 = np.int64(lw0 & U32_FFFF)
        l1 = np.int64(lw0 >> U32_16)
        l2 = np.int64(lw1 & U32_FFFF)
        l3 = np.int64(lw1 >> U32_16)
        rec[j, 0] |= last_seen[l0]
        rec[j, 1] = last_seen[l1]
        rec[j, 2] = last_seen[l2]
        rec[j, 3] = last_seen[l3]
        jj = np.uint32(j)
        last_seen[l0] = jj
        last_seen[l1] = jj
        last_seen[l2] = jj
        last_seen[l3] = jj
    # the number of flippable vertices per block of 64 operators, as a Fenwick tree
    n_blocks = fenwick.shape[0] - 1
    fenwick[0] = 0
    for block in range(n_blocks):
        fenwick[block + 1] = np.int32(count_bits(fp_mask[block]))
    for i in range(1, n_blocks + 1):
        parent = i + (i & (-i))
        if parent <= n_blocks:
            fenwick[parent] += fenwick[i]
    return n, n_fp, most, ok


@njit(cache=True, inline="always")
def count_bits(m):
    """The number of set bits of the 64-bit word m."""
    m = m - ((m >> U64_1) & np.uint64(0x5555555555555555))
    m = (m & np.uint64(0x3333333333333333)) + ((m >> np.uint64(2)) & np.uint64(0x3333333333333333))
    m = (m + (m >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (m * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(cache=True)
def fenwick_add(fenwick, block, delta):
    """Add delta to the count of `block`."""
    i = block + 1
    n = fenwick.shape[0] - 1
    while i <= n:
        fenwick[i] += delta
        i += i & (-i)


@njit(cache=True)
def kth_flippable_vertex(fenwick, fp_mask, k):
    """The operator number of the k-th flippable vertex (k >= 1)."""
    n = fenwick.shape[0] - 1
    block = 0
    step = 1
    while step * 2 <= n:
        step *= 2
    remaining = k
    while step > 0:  # the block in which the running count reaches k
        if block + step <= n and fenwick[block + step] < remaining:
            block += step
            remaining -= fenwick[block]
        step //= 2
    m = fp_mask[block]
    r = np.uint64(remaining)
    bit = np.uint64(0)
    shift = np.uint64(32)
    while shift > 0:  # the remaining-th set bit of the block, by halving
        low = m & ((U64_1 << shift) - U64_1)
        c = count_bits(low)
        if r > c:
            r -= c
            m >>= shift
            bit += shift
        else:
            m = low
        shift >>= U64_1
    return 64 * block + np.int64(bit)


@njit(cache=True)
def toggle_parity(parity, link, position):
    """A FLIP appears or disappears at `position` on `link`: toggle the parity of all later positions."""
    w = position >> 6
    r = position & 63
    if r < 63:
        parity[link, w] ^= U64_ALL << np.uint64(r + 1)
    for later in range(w + 1, parity.shape[1]):
        parity[link, later] ^= U64_ALL


# ---------------------------------------------------------------- cluster update


@njit(cache=True)
def cluster_update(
    state, ops_t, J, w_flippable, w_blocked, numbers, position, abort_lines, abort_prob,
    abort_steps, rec, slot_of, parity, first_op, n_ops, n_fp, fenwick, fp_mask, lines, wrap_links,
    marked, changed, touched, touched_fp, lined_link, lined_next, lined_position,
):  # fmt: skip
    """Propose a sweeping cluster from a random flippable vertex and accept or reject it."""
    # `lines[link]` (0 before and after): bit 0 = the new configuration differs from the old one on
    # this link at the current height, bit 1 = the link was lined at some time in this attempt
    # (`marked` lists those), bit 2 = the old value of a lined link (1 for +1). `lined_link` and
    # `lined_next` hold the lined links and the next operator on each. Returns 1 if accepted.
    n_fp_before = n_fp[0]
    if n_fp_before == 0:
        return 0
    start = kth_flippable_vertex(fenwick, fp_mask, random_below(n_fp_before, numbers, position) + 1)
    r0 = rec[start, 0]
    start_flip = (r0 & REC_FLIP) != 0
    log_J = np.log(J)
    log_w1 = np.log(w_flippable)
    log_w0 = np.log(w_blocked)
    log_weight_ratio = (log_w1 - log_J) if start_flip else (log_J - log_w1)
    rec[start, 0] = r0 ^ REC_FLIP  # the start vertex changes its type
    changed[0] = start
    n_changed = 1
    n_touched = 0  # vertices reached after the start vertex
    n_wrap = 0
    none = np.uint32(n_ops + 1)
    sf = np.int64(1) if start_flip else np.int64(0)
    # the four links of the start vertex get a line, with their old value above it
    nearest = none
    for k in range(4):
        next_k = rec[start, k] if k > 0 else (r0 & REC_NEXT)
        nearest = min(nearest, next_k)
        link = np.int64((rec[start, 6 + (k >> 1)] >> np.uint32(16 * (k & 1))) & U32_FFFF)
        pos = np.int64((rec[start, 4 + (k >> 1)] >> np.uint32(16 * (k & 1))) & U32_FFFF)
        old_below = (np.int64(1) if state[link] > 0 else np.int64(0)) ^ np.int64(
            (parity[link, pos >> 6] >> np.uint64(pos & 63)) & U64_1
        )
        lines[link] = np.int8(3 | ((old_below ^ sf) << 2))
        marked[k] = link
        lined_link[k] = link
        lined_next[k] = next_k
        lined_position[link] = k
    prefetch_row(rec, np.int64(nearest))
    n_lined = 4
    n_marked = 4
    choices_there = 0  # random 1/2 choices on the way there
    choices_back = 0  # ... and on the way back
    wrapped = False
    while True:
        # the next operator that a line reaches
        nearest = none
        for i in range(n_lined):
            nearest = min(nearest, lined_next[i])
        limit = np.uint32(start) if wrapped else np.uint32(n_ops)
        if nearest >= limit:
            if wrapped:  # back at the start vertex: the cluster did not close, no proposal
                for i in range(n_changed):
                    rec[changed[i], 0] ^= REC_FLIP
                for i in range(n_marked):
                    lines[marked[i]] = 0
                return 0
            wrapped = True
            # the lines cross the end of the string: they change `state` if the cluster is accepted
            for i in range(n_marked):
                link = marked[i]
                if lines[link] & 1:
                    wrap_links[n_wrap] = link
                    n_wrap += 1
            for i in range(n_lined):
                link = lined_link[i]
                lined_next[i] = first_op[link]
                lines[link] = np.int8(7 if state[link] > 0 else 3)
            continue
        second = none
        for i in range(n_lined):
            v = lined_next[i]
            second = min(second, v if v != nearest else none)
        prefetch_row(rec, np.int64(second))
        j = np.int64(nearest)
        wide_or_long = (abort_lines > 0 and n_lined > abort_lines) or (
            abort_steps > 0 and n_touched >= abort_steps
        )
        if wide_or_long and uniform(numbers, position) < abort_prob:  # give up this walk
            for i in range(n_changed):
                rec[changed[i], 0] ^= REC_FLIP
            for i in range(n_marked):
                lines[marked[i]] = 0
            return 0
        r0 = rec[j, 0]
        n1 = rec[j, 1]
        n2 = rec[j, 2]
        n3 = rec[j, 3]
        pw0 = rec[j, 4]
        pw1 = rec[j, 5]
        lw0 = rec[j, 6]
        lw1 = rec[j, 7]
        is_flip = (r0 & REC_FLIP) != 0
        tf = np.int64(1) if is_flip else np.int64(0)
        t = FLIP if is_flip else DIAG
        n0 = r0 & REC_NEXT
        l0 = np.int64(lw0 & U32_FFFF)
        l1 = np.int64(lw0 >> U32_16)
        l2 = np.int64(lw1 & U32_FFFF)
        l3 = np.int64(lw1 >> U32_16)
        p0 = np.int64(pw0 & U32_FFFF)
        p1 = np.int64(pw0 >> U32_16)
        p2 = np.int64(pw1 & U32_FFFF)
        p3 = np.int64(pw1 >> U32_16)
        v0 = np.int64(lines[l0])
        v1 = np.int64(lines[l1])
        v2 = np.int64(lines[l2])
        v3 = np.int64(lines[l3])
        # the old values below this vertex, as bits (1 for +1): a lined link carries its value,
        # the value of any other link follows from `state` and the parity at its position
        u0 = (np.int64(1) if state[l0] > 0 else np.int64(0)) ^ np.int64(
            (parity[l0, p0 >> 6] >> np.uint64(p0 & 63)) & U64_1
        )
        u1 = (np.int64(1) if state[l1] > 0 else np.int64(0)) ^ np.int64(
            (parity[l1, p1 >> 6] >> np.uint64(p1 & 63)) & U64_1
        )
        u2 = (np.int64(1) if state[l2] > 0 else np.int64(0)) ^ np.int64(
            (parity[l2, p2 >> 6] >> np.uint64(p2 & 63)) & U64_1
        )
        u3 = (np.int64(1) if state[l3] > 0 else np.int64(0)) ^ np.int64(
            (parity[l3, p3 >> 6] >> np.uint64(p3 & 63)) & U64_1
        )
        o0 = ((v0 >> 2) & 1) if (v0 & 1) != 0 else u0
        o1 = ((v1 >> 2) & 1) if (v1 & 1) != 0 else u1
        o2 = ((v2 >> 2) & 1) if (v2 & 1) != 0 else u2
        o3 = ((v3 >> 2) & 1) if (v3 & 1) != 0 else u3
        line_mask = (v0 & 1) | ((v1 & 1) << 1) | ((v2 & 1) << 2) | ((v3 & 1) << 3)
        # the likely next operator if the type stays: the nearest one on the lined links of q
        likely = none
        if (v0 & 1) != 0:
            likely = min(likely, n0)
        if (v1 & 1) != 0:
            likely = min(likely, n1)
        if (v2 & 1) != 0:
            likely = min(likely, n2)
        if (v3 & 1) != 0:
            likely = min(likely, n3)
        if likely < second:
            prefetch_row(rec, np.int64(likely))
        old_bits = o0 | (o1 << 1) | (o2 << 2) | (o3 << 3)
        new_bits = old_bits ^ line_mask
        # signs (+, +, -, -): U_p can act on the values (-, -, +, +) = 0b1100, U_p^dag on 0b0011
        old_below_fp = (old_bits == 12) | (old_bits == 3)
        new_below_fp = (new_bits == 12) | (new_bits == 3)
        same = line_mask == (15 if is_flip else 0)  # new state below = old state above
        cluster_ends = False
        if new_below_fp:
            if old_below_fp:
                if n_lined == 4:  # the four lines of q are the last ones: close them
                    new_type = DIAG if same else FLIP
                    cluster_ends = True
                else:  # other lines remain: keep the type and go on
                    new_type = t
            else:  # both types are possible: choose
                new_type = DIAG if uniform(numbers, position) < 0.5 else FLIP
                choices_there += 1
        else:  # only DIAG is possible, and the way back has a choice here if it was flippable
            new_type = DIAG
            choices_back += 1 if old_below_fp else 0
        log_w_new = (log_w1 if new_below_fp else log_w0) if new_type == DIAG else log_J
        log_w_old = (log_w1 if old_below_fp else log_w0) if t == DIAG else log_J
        log_weight_ratio += log_w_new - log_w_old
        touched[n_touched] = j
        touched_fp[n_touched] = 1 if (new_type == FLIP or new_below_fp) else 0
        n_touched += 1
        toggle = new_type != t
        tg = np.int64(1) if toggle else np.int64(0)
        if toggle:
            changed[n_changed] = j
            n_changed += 1
            rec[j, 0] = r0 ^ REC_FLIP
        # above this vertex a lined link of q moves on to its next operator. A change of type
        # toggles the line on all four links of q
        for k in range(4):
            if k == 0:
                link, v, o, next_k = l0, v0, o0, n0
            elif k == 1:
                link, v, o, next_k = l1, v1, o1, n1
            elif k == 2:
                link, v, o, next_k = l2, v2, o2, n2
            else:
                link, v, o, next_k = l3, v3, o3, n3
            was = v & 1
            if (was ^ tg) != 0:  # lined above this vertex
                if was:
                    i = lined_position[link]
                else:
                    if (v & 2) == 0:
                        marked[n_marked] = link
                        n_marked += 1
                    i = n_lined
                    lined_link[i] = link
                    lined_position[link] = i
                    n_lined += 1
                lines[link] = np.int8(3 | ((o ^ tf) << 2))
                lined_next[i] = next_k
            elif was:  # the line ends here
                lines[link] = 2
                i = lined_position[link]
                last = n_lined - 1
                moved = lined_link[last]
                lined_link[i] = moved
                lined_next[i] = lined_next[last]
                lined_position[moved] = i
                n_lined = last
        if cluster_ends:
            break
    for i in range(n_marked):
        lines[marked[i]] = 0
    # weight ratio, times the ratio of the ways to pick the start vertex and the 1/2 choices
    n_fp_after = n_fp_before + choices_there - choices_back
    ratio = (
        np.exp(log_weight_ratio) * n_fp_before / n_fp_after * 2.0 ** (choices_there - choices_back)
    )
    if uniform(numbers, position) < min(1.0, ratio):
        for i in range(n_changed):
            jj = changed[i]
            r0 = rec[jj, 0]
            ops_t[slot_of[jj]] = FLIP if (r0 & REC_FLIP) != 0 else DIAG
            for k in range(4):
                link = np.int64((rec[jj, 6 + (k >> 1)] >> np.uint32(16 * (k & 1))) & U32_FFFF)
                pos = np.int64((rec[jj, 4 + (k >> 1)] >> np.uint32(16 * (k & 1))) & U32_FFFF)
                toggle_parity(parity, link, pos)
        delta = 0
        for i in range(n_touched):
            jj = touched[i]
            r0 = rec[jj, 0]
            d = np.int64(touched_fp[i]) - (1 if (r0 & REC_FP) != 0 else 0)
            if d != 0:
                rec[jj, 0] = r0 ^ REC_FP
                delta += d
                fp_mask[jj >> 6] ^= U64_1 << np.uint64(jj & 63)
                fenwick_add(fenwick, jj >> 6, d)
        n_fp[0] += delta
        for i in range(n_wrap):
            link = wrap_links[i]
            state[link] = -state[link]
        return 1
    for i in range(n_changed):
        rec[changed[i], 0] ^= REC_FLIP
    return 0


@njit(cache=True, nogil=True)
def cluster_updates(
    n_attempts, state, ops_t, J, w_flippable, w_blocked, numbers, position, abort_lines,
    abort_prob, abort_steps, rec, slot_of, parity, first_op, n_ops, n_fp, fenwick, fp_mask, lines,
    wrap_links, marked, changed, touched, touched_fp, lined_link, lined_next, lined_position,
):  # fmt: skip
    """Make n_attempts cluster updates and return the number accepted."""
    accepted = 0
    for _ in range(n_attempts):
        accepted += cluster_update(
            state, ops_t, J, w_flippable, w_blocked, numbers, position, abort_lines, abort_prob,
            abort_steps, rec, slot_of, parity, first_op, n_ops, n_fp, fenwick, fp_mask, lines,
            wrap_links, marked, changed, touched, touched_fp, lined_link, lined_next,
            lined_position,
        )  # fmt: skip
    return accepted


# ---------------------------------------------------------------- the chain


class SSE:
    """One chain: `state` enters slot 0 of the string, which has n operators in M slots."""

    # The string is `ops_t` (type per slot) and `ops_p` (plaquette per slot, -1 for an empty slot).
    # `ops` gives it as a list of (type, plaquette).

    def __init__(
        self, lat, lam, state, seed=None, J=0.5, eps=0.25, M0=50,
        abort_lines=20, abort_prob=0.1, abort_steps=32,
    ):  # fmt: skip
        """Start from `state` with an empty string of M0 slots."""
        assert max(lat.n_plaq, lat.n_links) < 65536, "the index stores links in 16 bits"
        assert SIGNS.tolist() == [1, 1, -1, -1]
        self.lat = lat
        self.lam = lam
        self.J = J
        self.offset = (max(0.0, lam) + eps) * J  # added to H so that every weight is positive
        self.w_flippable = self.offset - lam * J  # weight of DIAG on a flippable plaquette
        self.w_blocked = self.offset  # weight of DIAG on any other plaquette
        self.abort_lines = abort_lines
        self.abort_prob = abort_prob
        self.abort_steps = abort_steps
        self.state = np.array(state, dtype=np.int8)
        self.ops_t = np.full(M0, IDENTITY, np.int64)
        self.ops_p = np.full(M0, -1, np.int64)
        self.n = 0
        self.numbers = NO_NUMBERS  # injected random numbers (tests), read from `position`
        self.position = np.zeros(1, np.int64)
        self.generator_state = None
        use_generator_of(self)
        seed_generator(int.from_bytes(os.urandom(4)) if seed is None else seed)

        n_links = lat.n_links
        self.links = np.ascontiguousarray(lat.plaq_links, dtype=np.int64)
        self.links4 = self.links.astype(np.uint16).ravel()
        self.n_fp = np.zeros(1, np.int64)  # number of flippable vertices
        self.link_count = np.zeros(n_links, np.int64)
        self.first_op = np.zeros(n_links, np.uint32)
        self.last_seen = np.zeros(n_links, np.uint32)
        self.parity = np.zeros((n_links, 2), np.uint64)
        self.lines = np.zeros(n_links, np.int8)
        self.wrap_links = np.zeros(n_links, np.int64)
        self.marked = np.zeros(n_links + 1, np.int64)
        self.lined_link = np.zeros(n_links, np.int32)
        self.lined_next = np.zeros(n_links, np.uint32)
        self.lined_position = np.zeros(n_links, np.int32)
        self.index_is_current = False
        self.allocate_index()

    @property
    def M(self):
        """The number of slots."""
        return len(self.ops_t)

    @property
    def ops(self):
        """The string as a list of (type, plaquette), one per slot."""
        return [(int(t), int(p)) for t, p in zip(self.ops_t, self.ops_p, strict=True)]

    @ops.setter
    def ops(self, ops):
        """Set the string from a list of (type, plaquette), one per slot."""
        self.ops_t = np.array([t for t, _ in ops], np.int64)
        self.ops_p = np.array([p for _, p in ops], np.int64)
        self.n = int(np.sum(self.ops_t != IDENTITY))
        self.index_is_current = False

    def inject(self, numbers):
        """Use `numbers` as the random numbers from now on (for tests), or the generator again if the list is empty."""
        self.numbers = np.array(numbers, dtype=float)
        self.position[0] = 0

    # ---------------------------------------------------------------- operators

    def weight(self, s, t, p):
        """Matrix element of the operator (t, p) on the state s."""
        if t == IDENTITY:
            return 1.0
        flippable = self.lat.is_flippable(s, p)
        if t == DIAG:
            return self.w_flippable if flippable else self.w_blocked
        return self.J if flippable else 0.0

    # ---------------------------------------------------------------- the string

    def is_valid(self):
        """Whether every operator has a positive weight and the last state equals the first."""
        return bool(string_is_valid(self.state, self.ops_t, self.ops_p, self.links))

    def k_vector(self):
        """Per plaquette the number of U_p minus the number of U_p^dag in the string."""
        return string_k_vector(self.state, self.ops_t, self.ops_p, self.links, self.lat.n_plaq)

    def operator_states(self):
        """The state entering each operator of the string, one row per operator."""
        return string_operator_states(self.state, self.ops_t, self.ops_p, self.links)

    def mode(self, w):
        """sum_l w_l s_l on the state entering each operator of the string."""
        w = np.asarray(w, dtype=complex)
        return string_mode(self.state, self.ops_t, self.ops_p, self.links, w)

    def energy(self, beta):
        """Estimator of <H> for the current string: -n / beta plus the offset that was added."""
        return -self.n / beta + self.lat.n_plaq * self.offset

    # ---------------------------------------------------------------- sweep

    def sweep(self, beta):
        """One Monte Carlo sweep: diagonal update, N_p cluster attempts, time shift, cutoff."""
        self.diagonal_update(beta)
        self.cluster_updates(self.lat.n_plaq)
        self.time_shift_update()
        self.adjust_cutoff()

    def diagonal_update(self, beta):
        """Go through the string once, inserting a DIAG in empty slots and removing DIAGs."""
        use_generator_of(self)
        self.build_index(beta, update=True)

    def allocate_index(self):
        """Make the arrays of the index whose size depends on the number of slots."""
        M = self.M
        # rows n .. M + 1 of rec are only asked for by prefetch_row. A row is 32 bytes and
        # starts on a multiple of 64 bytes, so that no record lies across two cache lines
        buffer = np.empty((M + 2) * 8 + 16, np.uint32)
        offset = (-buffer.ctypes.data // 4) % 16
        self.rec = buffer[offset : offset + (M + 2) * 8].reshape(M + 2, 8)
        n_blocks = (M + 2 + 63) // 64
        self.slot_of = np.empty(M + 2, np.uint32)
        self.fenwick = np.zeros(n_blocks + 1, np.int32)
        self.fp_mask = np.zeros(n_blocks, np.uint64)
        self.changed = np.empty(M + 1, np.int64)
        self.touched = np.empty(M + 1, np.int64)
        self.touched_fp = np.empty(M + 1, np.uint8)
        self.index_M = M  # the number of slots these arrays were made for

    def build_index(self, beta=0.0, update=False):
        """Build the index of the string, with the diagonal update if `update`."""
        if self.index_M != self.M:
            self.allocate_index()
        while True:
            self.n, n_fp, most, ok = diagonal_update_and_index(
                self.state, self.ops_t, self.ops_p, self.n, beta, self.lat.n_plaq, self.links4,
                self.w_flippable, self.w_blocked, self.numbers, self.position, update, self.rec,
                self.slot_of, self.link_count, self.first_op, self.parity, self.fenwick,
                self.fp_mask, self.last_seen,
            )  # fmt: skip
            if ok:
                break
            # the positions 0 .. most on a link need most + 1 bits
            self.parity = np.zeros((self.lat.n_links, most // 64 + 1), np.uint64)
            update = False
        assert most < 65536, "the index stores positions in 16 bits"
        self.n_fp[0] = n_fp
        self.index_is_current = True

    def cluster_updates(self, n_attempts):
        """Make n_attempts sweeping-cluster updates and return the number accepted."""
        use_generator_of(self)
        if not self.index_is_current:
            self.build_index()
        return cluster_updates(
            n_attempts, self.state, self.ops_t, self.J, self.w_flippable, self.w_blocked,
            self.numbers, self.position, self.abort_lines, self.abort_prob, self.abort_steps,
            self.rec, self.slot_of, self.parity, self.first_op, self.n, self.n_fp, self.fenwick,
            self.fp_mask, self.lines, self.wrap_links, self.marked, self.changed, self.touched,
            self.touched_fp, self.lined_link, self.lined_next, self.lined_position,
        )  # fmt: skip

    # ---------------------------------------------------------------- time shift and cutoff

    def time_shift_update(self):
        """Rotate the string by a random number of slots."""
        use_generator_of(self)
        rotate_string(self.state, self.ops_t, self.ops_p, self.links, self.numbers, self.position)
        self.index_is_current = False

    def adjust_cutoff(self):
        """Append empty slots when more than 70 % of the string is filled."""
        if self.n > 0.7 * self.M:
            longer = max(self.M + 1, self.n + self.n // 3)
            extra = longer - self.M
            self.ops_t = np.concatenate([self.ops_t, np.full(extra, IDENTITY, np.int64)])
            self.ops_p = np.concatenate([self.ops_p, np.full(extra, -1, np.int64)])
            self.index_is_current = False
