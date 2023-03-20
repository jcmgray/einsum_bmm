import pytest


def random_single_eq(
    min_dim=1,
    max_dim=8,
    d_low=2,
    d_high=5,
    unique_axes=False,
    seed=None,
):
    import opt_einsum as oe
    import numpy as np

    rng = np.random.default_rng(seed)

    ndim = rng.integers(min_dim, max_dim + 1)
    num_indices = max(1, rng.poisson(1.5 * ndim))
    out_ndim = rng.poisson(0.5 * ndim)

    all_ix = [oe.get_symbol(i) for i in range(num_indices)]
    size_dict = dict(
        zip(all_ix, rng.integers(d_low, d_high + 1, size=num_indices))
    )
    term = "".join(
        rng.choice(all_ix, size=ndim, replace=not unique_axes)
    )
    shape = tuple(size_dict[ix] for ix in term)

    used = sorted(set(term))
    out_ndim = min(out_ndim, len(used))
    out = "".join(rng.choice(used, size=out_ndim, replace=False))

    eq = f"{term}->{out}"

    return eq, shape


@pytest.mark.parametrize('seed', range(1000))
def test_random_single_eq(seed):
    import numpy as np
    from einsum_bmm import einsum
    eq, shape = random_single_eq(seed=seed)
    x = np.random.randn(*shape)

    y1 = einsum(eq, x)
    y2 = np.einsum(eq, x)
    assert np.allclose(y1, y2)



def random_pair_eq(
    min_dim=1,
    max_dim=8,
    d_low=2,
    d_high=5,
    unique_axes=False,
    seed=None,
):
    import opt_einsum as oe
    import numpy as np

    rng = np.random.default_rng(seed)

    a_ndim = rng.integers(min_dim, max_dim + 1)
    b_ndim = rng.integers(min_dim, max_dim + 1)
    num_indices = max(1, a_ndim + b_ndim)
    out_ndim = rng.poisson(0.5 * (a_ndim + b_ndim))

    all_ix = [oe.get_symbol(i) for i in range(num_indices)]
    size_dict = dict(
        zip(all_ix, rng.integers(d_low, d_high + 1, size=num_indices))
    )
    a_term = "".join(
        rng.choice(all_ix, size=a_ndim, replace=not unique_axes)
    )
    b_term = "".join(
        rng.choice(all_ix, size=b_ndim, replace=not unique_axes)
    )
    shape_a = tuple(size_dict[ix] for ix in a_term)
    shape_b = tuple(size_dict[ix] for ix in b_term)

    used = sorted(set(a_term + b_term))
    out_ndim = min(out_ndim, len(used))
    out = "".join(rng.choice(used, size=out_ndim, replace=False))

    eq = f"{a_term},{b_term}->{out}"

    return eq, shape_a, shape_b


@pytest.mark.parametrize('seed', range(1000))
def test_random_pair_eq(seed):
    import numpy as np
    from einsum_bmm import einsum
    eq, shape_a, shape_b = random_pair_eq(seed=seed)
    a = np.random.randn(*shape_a)
    b = np.random.randn(*shape_b)

    y1 = einsum(eq, a, b)
    y2 = np.einsum(eq, a, b)
    assert np.allclose(y1, y2)
