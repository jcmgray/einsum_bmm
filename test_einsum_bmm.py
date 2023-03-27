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


def random_tensordot(
    min_dim=1,
    max_dim=6,
    d_low=2,
    d_high=4,
    seed=None,
):
    import numpy as np

    rng = np.random.default_rng(seed)

    a_ndim = rng.integers(min_dim, max_dim + 1)
    b_ndim = rng.integers(min_dim, max_dim + 1)
    num_contracted = rng.integers(0, min(a_ndim, b_ndim) + 1)
    a_shape = rng.integers(d_low, d_high + 1, size=a_ndim)
    b_shape = rng.integers(d_low, d_high + 1, size=b_ndim)


    if rng.random() < 0.1:
        # single int specification
        axes = num_contracted
        if axes > 0:
            b_shape[:axes] = a_shape[-axes:]
    else:
        a_axes = rng.choice(a_ndim, size=num_contracted, replace=False)
        b_axes = rng.choice(b_ndim, size=num_contracted, replace=False)
        axes = (a_axes, b_axes)
        a_shape[a_axes] = b_shape[b_axes]

    return a_shape, b_shape, axes


@pytest.mark.parametrize('seed', range(100))
def test_random_tensordot(seed):
    import numpy as np
    from einsum_bmm import tensordot

    a_shape, b_shape, axes = random_tensordot(seed=seed)

    a = np.random.randn(*a_shape)
    b = np.random.randn(*b_shape)

    y1 = tensordot(a, b, axes=axes)
    y2 = np.tensordot(a, b, axes=axes)
    assert np.allclose(y1, y2)


def eq_to_shapes(eq, d_min, d_max, seed=None):
    import numpy as np
    rng = np.random.default_rng(seed)
    sizes = {}
    lhs = eq.split('->')[0]
    terms = lhs.split(',')
    shapes = []
    for term in terms:
        shape = ()
        for ix in term:
            if ix not in sizes:
                sizes[ix] = rng.integers(d_min, d_max + 1)
            d = sizes[ix]
            shape += (d,)
        shapes.append(shape)
    return shapes


@pytest.mark.parametrize('eq', [
    'ab->ab',
    'ab->ba',
    'ab,bc->ac',
    'a,ab->b',
    'ab,b->a',
    'a,a->',
    'ab,ab->',
    'ab,ab->',
    'abc,abc->abc',
    ',->',
])
@pytest.mark.parametrize('seed', range(10))
def test_specific_cases(eq, seed):
    import numpy as np
    from einsum_bmm import einsum

    rng = np.random.default_rng(seed)
    shapes = eq_to_shapes(eq, 2, 5, seed=seed)
    arrays = [rng.random(shape) for shape in shapes]

    y1 = einsum(eq, *arrays)
    y2 = np.einsum(eq, *arrays)
    assert np.allclose(y1, y2)
