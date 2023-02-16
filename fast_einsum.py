import math
import functools
import autoray as ar


@functools.lru_cache(2**12)
def _sanitize_equation(eq):
    """Get the input and output indices of an equation, computing the output
    implicitly as the sorted sequence of every index that appears exactly once
    if it is not  provided.
    """
    # remove spaces
    eq = eq.replace(" ", "")

    if "->" not in eq:
        lhs = eq
        tmp_subscripts = lhs.replace(",", "")
        out = "".join(
            # sorted sequence of indices
            s
            for s in sorted(set(tmp_subscripts))
            # that appear exactly once
            if tmp_subscripts.count(s) == 1
        )
    else:
        lhs, out = eq.split("->")
    return lhs, out


@functools.lru_cache(2**12)
def _parse_einsum_single(eq, shape):
    """Cached parsing of a single term einsum equation into the necessary
    sequence of arguments for axes diagonals, sums, and transposes.
    """
    lhs, out = _sanitize_equation(eq)

    # parse each index
    need_to_diag = []
    need_to_sum = []
    seen = set()
    for ix in lhs:
        if ix in need_to_diag:
            continue
        if ix in seen:
            need_to_diag.append(ix)
            continue
        seen.add(ix)
        if ix not in out:
            need_to_sum.append(ix)

    # first handle diagonal reductions
    if need_to_diag:
        diag_sels = []
        sizes = dict(zip(lhs, shape))
        while need_to_diag:
            ixd = need_to_diag.pop()
            dinds = tuple(range(sizes[ixd]))

            # construct advanced indexing object
            selector = tuple(dinds if ix == ixd else slice(None) for ix in lhs)
            diag_sels.append(selector)

            # after taking the diagonal what are new indices?
            ixd_contig = ixd * lhs.count(ixd)
            if ixd_contig in lhs:
                # contig axes, new axis is at same position
                lhs = lhs.replace(ixd_contig, ixd)
            else:
                # non-contig, new axis is at beginning
                lhs = ixd + lhs.replace(ixd, "")
    else:
        diag_sels = None

    # then sum reductions
    if need_to_sum:
        sum_axes = tuple(map(lhs.index, need_to_sum))
        for ix in need_to_sum:
            lhs = lhs.replace(ix, "")
    else:
        sum_axes = None

    # then transposition
    if lhs == out:
        perm = None
    else:
        perm = tuple(lhs.index(ix) for ix in out)

    return diag_sels, sum_axes, perm


@functools.lru_cache(2**12)
def parse_double_eq(eq, shape_a, shape_b):
    """Cached parsing of a two term einsum equation into the necessary
    sequence of arguments for contracttion via batched matrix multiplication.
    The steps we need to specify are:

        1. Remove repeated and trivial indices from the left and right terms,
           and transpose them, done as a single einsum.
        2. Fuse the remaining indices so we have two 3D tensors.
        3. Perform the batched matrix multiplication.
        4. Unfuse the output to get the desired final index order.


    """
    lhs, out = _sanitize_equation(eq)
    lterm, rterm = lhs.split(",")

    bat_inds = []  # appears L, R, O
    con_inds = []  # appears L, R, .
    left_keep = []  # appears L, ., O
    left_only = []  # appears L, ., .
    right_keep = []  # appears ., R, O
    right_only = []  # appears ., R, .
    sizes = {}

    # parse left term
    seen = set()
    for ix, d in zip(lterm, shape_a):
        # set or check size
        if sizes.setdefault(ix, d) != d:
            raise ValueError(
                f"Index {ix} has mismatched sizes {sizes[ix]} and {d}."
            )

        if ix in seen:
            continue
        seen.add(ix)

        if ix in rterm:
            if ix in out:
                bat_inds.append(ix)
            else:
                con_inds.append(ix)
        elif ix in out:
            left_keep.append(ix)
        else:
            left_only.append(ix)

    # parse right term
    seen.clear()
    for ix, d in zip(rterm, shape_b):
        # set or check size
        if sizes.setdefault(ix, d) != d:
            raise ValueError(
                f"Index {ix} has mismatched sizes {sizes[ix]} and {d}."
            )

        if ix in seen:
            continue
        seen.add(ix)

        if ix not in lterm:
            if ix in out:
                right_keep.append(ix)
            else:
                right_only.append(ix)

    # take diagonal, remove any trivial axes and transpose left
    desired_a = "".join((*bat_inds, *left_keep, *con_inds))
    if lterm != desired_a:
        eq_a = f"{lterm}->{desired_a}"
    else:
        eq_a = None

    # take diagonal, remove any trivial axes and transpose right
    desired_b = "".join((*bat_inds, *con_inds, *right_keep))
    if rterm != desired_b:
        eq_b = f"{rterm}->{desired_b}"
    else:
        eq_b = None

    # then we want to permute the matmul produced output:
    out_produced = "".join((*bat_inds, *left_keep, *right_keep))
    perm_ab = tuple(out_produced.index(ix) for ix in out)
    if perm_ab == tuple(range(len(perm_ab))):
        perm_ab = None

    # then we want to reshape
    if bat_inds:
        lgroups = (bat_inds, left_keep, con_inds)
        rgroups = (bat_inds, con_inds, right_keep)
        ogroups = (bat_inds, left_keep, right_keep)
    else:
        # avoid size 1 batch dimension if no batch indices
        lgroups = (left_keep, con_inds)
        rgroups = (con_inds, right_keep)
        ogroups = (left_keep, right_keep)

    if any(len(group) != 1 for group in lgroups):
        new_shape_a = tuple(
            math.prod(sizes[ix] for ix in ix_group) for ix_group in lgroups
        )
    else:
        new_shape_a = None

    if any(len(group) != 1 for group in rgroups):
        new_shape_b = tuple(
            math.prod(sizes[ix] for ix in ix_group) for ix_group in rgroups
        )
    else:
        new_shape_b = None

    if any(len(group) != 1 for group in ogroups):
        new_shape_ab = tuple(
            sizes[ix] for ix_group in ogroups for ix in ix_group
        )
    else:
        new_shape_ab = None

    return (
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
    )


def _einsum_single(eq, x, backend=None):
    """Einsum on a single tensor, via three steps: diagonal selection
    (via advanced indexing), axes summations, transposition. The logic for each
    is cached based on the equation and array shape, and each step is only
    performed if necessary.
    """
    diag_sels, sum_axes, perm = _parse_einsum_single(eq, x.shape)

    if diag_sels is not None:
        # diagonal reduction via advanced indexing
        # e.g ababbac->abc
        for selector in diag_sels:
            x = x[selector]

    if sum_axes is not None:
        # trivial removal of axes via summation
        # e.g. abc->c
        x = ar.do("sum", x, sum_axes, like=backend)

    if perm is not None:
        # transpose to desired output
        # e.g. abc->cba
        x = ar.do("transpose", x, perm, like=backend)

    return x


def einsum(eq, a, b=None, backend=None):
    """Perform arbitrary single and pairwise einsums using only `matmul`,
    `transpose`, `reshape` and `sum`.  The logic for each is cached based on
    the equation and array shape, and each step is only performed if necessary.
    """
    if b is None:
        return _einsum_single(eq, a, backend=backend)

    # ensure we can cache on shapes
    a_shape = tuple(map(int, a.shape))
    b_shape = tuple(map(int, b.shape))
    (
        eq_a,
        eq_b,
        new_shape_a,
        new_shape_b,
        new_shape_ab,
        perm_ab,
    ) = parse_double_eq(eq, a_shape, b_shape)

    # prepare left
    if eq_a is not None:
        # diagonals, sums, and tranpose
        a = _einsum_single(eq_a, a)
    if new_shape_a is not None:
        a = ar.do("reshape", a, new_shape_a, like=backend)

    # prepare right
    if eq_b is not None:
        # diagonals, sums, and tranpose
        b = _einsum_single(eq_b, b)
    if new_shape_b is not None:
        b = ar.do("reshape", b, new_shape_b, like=backend)

    # do the contraction!
    ab = ar.do("matmul", a, b, like=backend)

    # prepare the output
    if new_shape_ab is not None:
        ab = ar.do("reshape", ab, new_shape_ab, like=backend)
    if perm_ab is not None:
        ab = ar.do("transpose", ab, perm_ab, like=backend)

    return ab


# # enable in opt_einsum:
# import opt_einsum as oe
# oe.backends.dispatch._cached_funcs['einsum', 'numpy'] = einsum
# oe.backends.dispatch._cached_funcs['einsum', 'cupy'] = einsum
# oe.backends.dispatch._cached_funcs['einsum', 'torch'] = einsum
