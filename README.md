# libsepal
> Semantic (meaning) axis vector store — two-stage ANN search for the recall kernel.

Store float vectors with per-blob dimensions, then query by meaning: a
Hamming prefilter over a sign-bit sketch picks the top-m candidates, an exact
cosine rerank keeps the best-k. Kernel-form adapters (`sepal_fill_approx`,
`sepal_rank`) plug the meaning axis straight into `rec_set_t` joins, and the
fill declares its approximate-ness so downstream recall stays honest.

## Key Features

- **Dim-tagged blobs (VEC1)**: every vector stores its own exact-stage dim
  (`min(full_dim, 256)` matryoshka prefix) and a full-dim sign sketch —
  1,136 bytes at 768/256 instead of ~7 KB of text
- **Two-stage ANN**: Hamming popcount prefilter → exact cosine rerank,
  `m = 10×k` by default, `min_sim` filter at rerank
- **Kernel-form fill**: `sepal_fill_approx()` streams top-m refs into a
  sealed `rec_set_t` and marks it `REC_SET_APPROX` with an owed recall bound
  (`m/n`); joins propagate the flag via the kernel extension
- **Exact rerank callback**: `sepal_rank()` scores one ref (a `rec_score_fn`
  shape: 0 ok, nonzero when missing / dim-mismatched / below `min_sim`)
- **File persistence**: optional sidecar via libqmap; `sepal_close()` saves
  (never closes — the file would truncate to 0, same invariant as mm);
  the vector count is restored from the file on open
- **Caller-opaque refs**: refs are `rec_ref_t` (u64) passed at put time;
  the store never maps refs to schemas

## Quick Start

```c
#include <ttypt/sepal.h>

sepal_vecstore_t *vs = sepal_open(NULL, NULL);   /* memory-only */
sepal_put(vs, 1, red, 4);                        /* ref 1 ← 4-dim vector */

float q[4] = { 0.9f, 0.1f, 0.0f, 0.0f };
sepal_hit_t hits[2];
size_t n = sepal_search(vs, q, 4, 2, 0.0f, 0, hits);  /* m=0 → 10×k */
for (size_t i = 0; i < n; i++)
    printf("ref=%llu score=%.4f\n",
           (unsigned long long)hits[i].ref, hits[i].score);
sepal_close(vs);
```

See `examples/` (`basic`, `twostage`, `persistence`, `kernel_form`).

## VEC1 blob format

Little-endian, binary-only (legacy text blobs start with a digit, so they
can never collide):

```
off  size  field
0    4     magic 'V','E','C','1'
4    2     version (1)
6    2     dim           (exact-stage dim = min(full_dim, 256))
8    2     full_dim      (sketch source dim)
10   2     sketch_words  (W = ceil(full_dim/64))
12   4     norm          (float32, ||v[0..dim)|| frozen at put)
16   8W    sketch        (W u64 sign words over the FULL-dim vector)
16+8W 4·dim vec          (float32 truncated prefix)
total 16+8W+4·dim → 1,136 B at 768/256
```

`sepal_blob_len()` / `sepal_blob_hdr()` / `sepal_blob_put()` expose the
format; garbage, truncation, wrong magic/version, or inconsistent fields
always decode to 0 — never a misread.

## API (see `include/ttypt/sepal.h` for the contract)

Store: `sepal_open` / `sepal_close` / `sepal_put` / `sepal_del` /
`sepal_get` (never truncates: over-max → full dim, under-max → 0) /
`sepal_dim` / `sepal_full_dim` / `sepal_n`.
Math: `sepal_cosine` (0 on zero norm), `sepal_sketch` (sign bits, padding zeroed).
Search: `sepal_search` (two-stage; `m==0` → 10×k capped at N; returns hits
written ≤ k, best-first, ties by ascending ref).
Kernel: `sepal_fill_approx` (m==0 → full pool; additive over existing refs),
`sepal_rank` + `sepal_rank_ctx`.

## Building & testing

```sh
make          # builds lib/libsepal.so (needs ../mk + the site-tree libqmap)
make test     # unit + integration + property + stress (see tests/TESTING.md)
make bench    # two-stage vs brute force (records study §12.4)
make -C tests valgrind | asan | ubsan
```

Binaries carry no rpath: the Makefiles export `LD_LIBRARY_PATH` (own `lib/`
plus the site-tree `libqmap` that carries the kernel approximation flag).
Until `libqmap` is reinstalled system-wide with the extension, consumers
link `-L<site>/external/libqmap/lib` and run with the same path prefix.

## Limitations

- Exact brute force is a test-side reference, not a library feature —
  exactness is a kernel property, not a sepal feature.
- Query dim must be ≥ the stored exact dim for a candidate to be ranked
  (ranked on the truncated prefix); the sketch window compares the first
  `min(Wq, Wstored)` words.
- `SEPAL_EXACT_DIM` (256) is a default baked into each blob's dims, not
  a store-wide config; no per-store dim override in 0.1.0.
- Norm is frozen at put: mutating floats in place is not supported —
  re-put the ref instead.
