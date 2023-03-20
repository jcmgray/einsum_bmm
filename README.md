# `einsum_bmm`

This repository provides an `einsum` function implemented via **batch matrix
multiply**.

1. This *can* be much faster than the raw `numpy.einsum` function, especially
   for large and high dimensional contractions.
2. It can also be used to enable `einsum` for any backend that provides only
   `tranpose`, `reshape` and `matmul`.

Notes:

* It currently only supports 1 or 2 terms, a library such as `opt_einsum` or
  `cotengra` should be used to dispatch many term contractions to a pairwise
  ordering in conjuction with this `einsum_bmm`.
