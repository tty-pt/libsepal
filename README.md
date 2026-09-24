# libsepal

[![C99](https://img.shields.io/badge/C-C99-555?logo=c)](#)
[![BSD-2-Clause](https://img.shields.io/badge/License-BSD--2--Clause-blue)](#)
[![semantic-ANN](https://img.shields.io/badge/semantic-ANN-FF6B35)](#)

> Semantic (meaning) axis vector store — two-stage ANN search for the recall kernel.

Store float vectors with per-blob dimensions, then query by meaning: a
Hamming prefilter over a sign-bit sketch picks the top-m candidates, an exact
cosine rerank keeps the best-k. Kernel-form adapters (`sepal_fill_approx`,
`sepal_rank`) plug the meaning axis straight into `rec_set_t` joins, and the
fill declares its approximate-ness so downstream recall stays honest.

## Contents

- [Features](#features)
- [Install](#install)
- [Build from source](#build-from-source)
- [Quickstart](#quickstart)
- [VEC1 blob format](#vec1-blob-format)
- [API overview](#api-overview)
- [Performance](#performance)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [Testing](#testing)
- [License](#license)

## Features

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
- **File persistence**: optional sidecar via libcorm; `sepal_close()` saves
  (never closes — the file would truncate to 0, same invariant as mm);
  the vector count is restored from the file on open
- **Flat sketch index**: stage-1 scans contiguous `idx_sk` rows
  (~1 MB at N=10k, L2-resident) instead of walking full blobs — ~3×
  faster prefilter at N=10k; rebuilt at open, maintained on put/del,
  `sepal_index_validate()` checks internal consistency
- **Frozen-norm rerank**: stage-2 scores from one fused dot pass using
  the put-frozen blob norm + a once-per-query prefix-norm table
  (semantically equal to `sepal_cosine` within 1e-6)
- **AVX2 fast path**: sign-bit sketch via `vmovmskps`; Hamming prefilter
  via 256-bit XOR→nibble-LUT popcount; cosine loops auto-vectorized with
  FMA; scalar fallback retained for non-AVX2 targets
- **Caller-opaque refs**: refs are `rec_ref_t` (u32) passed at put time;
  the store never maps refs to schemas

## Install

Prebuilt packages are distributed on tty.pt for Linux (APT / Alpine / Arch /
Fedora-RHEL), macOS (Homebrew), Windows (winget / MSYS2), and OpenBSD.
Follow the [installation instructions](
https://github.com/tty-pt/ci/blob/main/docs/install.md) and use
**libsepal** as the package name.

## Build from source

The library builds with a plain `make` (the shared [`mk` include.mk](
https://github.com/tty-pt/mk)):

```sh
make                  # builds lib/libsepal.so
make test             # run the in-tree test suite
sudo make install     # lib + headers + sepal.pc -> $(PREFIX), default /usr/local
```

Link it from your own C code:

```sh
cc my_app.c $(pkg-config --cflags --libs sepal)
```

**Dependencies:** `libcorm`, `libqsys`, `libxxhash`. The build expects
`../mk` and, until `libcorm` is reinstalled system-wide with the kernel
approximation flag, a site-tree `libcorm`.

## Quickstart

```c
#include <ttypt/sepal.h>

sepal_vecstore_t *vs = sepal_open(NULL, NULL);   /* memory-only */
sepal_put(vs, 1, red, 4);                        /* ref 1 ← 4-dim vector */

float q[4] = { 0.9f, 0.1f, 0.0f, 0.0f };
sepal_hit_t hits[2];
size_t n = sepal_search(vs, q, 4, 2, 0.0f, 0, hits);  /* m=0 → 10×k */
for (size_t i = 0; i < n; i++)
    printf("ref=%u score=%.4f\n",
           hits[i].ref, hits[i].score);
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

## API overview

The contract lives in `include/ttypt/sepal.h`.

Store: `sepal_open` / `sepal_close` / `sepal_put` / `sepal_del` /
`sepal_get` (never truncates: over-max → full dim, under-max → 0) /
`sepal_dim` / `sepal_full_dim` / `sepal_n`.
Math: `sepal_cosine` (0 on zero norm), `sepal_sketch` (sign bits, padding zeroed).
Search: `sepal_search` (two-stage; `m==0` → 10×k capped at N; returns hits
written ≤ k, best-first, ties by ascending ref).
Kernel: `sepal_fill_approx` (m==0 → full pool; additive over existing refs),
`sepal_rank` + `sepal_rank_ctx`.

## Performance (medians-of-3 at N=10000; single clean run at N=100/1000, `bench_search`, k=10, m=100)

| N | dim | exact µs | prefilter µs | rerank µs | total µs | full@10 | mean r@10 |
|---|---|---|---|---|---|---|---|
| 100 | 384 | 60 | 13 | 46 | 51 | 1.000 | 1.000 |
| 1000 | 384 | 869 | 49 | 155 | 81 | 0.030 | 0.676 |
| 10000 | 384 | 35709 | 235 | 213 | 320 | 0.000 | 0.366 |
| 100 | 768 | 64 | 18 | 56 | 56 | 1.000 | 1.000 |
| 1000 | 768 | 1121 | 55 | 94 | 86 | 0.000 | 0.463 |
| 10000 | 768 | 35828 | 279 | 210 | 372 | 0.000 | 0.164 |

Absolute numbers move 2–3× with machine load (even `exact_us`, which is
fixed work, swings 40k–50k) — treat single runs as noisy; medians-of-3
at N=10000 are the reliable row. Vs the pre-AVX2 shipped config (bisect D):
prefilter 3.3×/4.4×, rerank 1.44×/1.63×, total 2.9×/4.3× faster at
384/768 dim. Recall columns are unchanged by every tier (D1 semantic rule).

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

## Documentation

- [examples/README.md](./examples/README.md) — runnable examples
- [tests/TESTING.md](./tests/TESTING.md) — test layout
- [CHANGELOG.md](./CHANGELOG.md) — version history
- [include/ttypt/sepal.h](./include/ttypt/sepal.h) — full API contract

## Testing

```sh
make test      # unit + integration + property + stress (see tests/TESTING.md)
make -C tests valgrind | asan | ubsan
```

## License

BSD 2-Clause License. Copyright (c) 2025, tty-pt. See `LICENSE`.