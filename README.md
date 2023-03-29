# `einsum_bmm`

This repository provides an `einsum` (and `tensordot`) function implemented via **batch matrix
multiply**.

1. This *can* be much faster than the raw `numpy.einsum` function, especially
   for large and high dimensional contractions.
2. It can also be used to enable `einsum` for any backend that provides only
   `tranpose`, `reshape` and `matmul`.

The implementation is achieved by grouping indices according to the following classification:

<img src="https://user-images.githubusercontent.com/8982598/228432891-595c88af-cb81-443e-9cf3-5eda86db01b2.png" alt="Schematic" width="500" title="Einsum Schematic">

1. Summed indices are trivially removed.
2. A and B and then transposed and reshaped for batched matrix multiplication
3. The output is reshaped and transposed

Each of these steps only occurs if necessary. There are slight specializations for both pure multiplication and no batch indices.

Notes:

* It currently only supports 1 or 2 terms, a library such as `opt_einsum` or
  `cotengra` should be used to dispatch many term contractions to a pairwise
  ordering in conjuction with this `einsum_bmm`.

