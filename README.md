<p align="center">
  <img src="assets/logo.png" alt="libsepal logo" width="220" />
</p>

# libsepal

[![C99](https://img.shields.io/badge/C-C99-555?logo=c)](#)
[![BSD-2-Clause](https://img.shields.io/badge/License-BSD--2--Clause-blue)](#)
[![semantic-ANN](https://img.shields.io/badge/semantic-ANN-FF6B35)](#)
[![Recall Kernel Meaning Axis](https://img.shields.io/badge/axis-meaning-blueviolet)](#)

> Ultra-fast semantic vector store with two-stage ANN search (Hamming sign-bit sketch prefilter + exact cosine rerank) in pure C. Canonical header is `<ttypt/sepal.h>`.

---

## 1. Quickstart

Store vector embeddings and perform nearest-neighbor semantic search in under 15 lines:

```c
#include <stdio.h>
#include <ttypt/sepal.h>

int main(void) {
    // Open in-memory semantic vector store
    sepal_vecstore_t *vs = sepal_open(NULL, NULL);

    float vec_a[4] = {0.9f, 0.1f, 0.0f, 0.0f};
    float vec_b[4] = {0.8f, 0.2f, 0.1f, 0.0f};
    sepal_put(vs, 1, vec_a, 4); // Ref 1
    sepal_put(vs, 2, vec_b, 4); // Ref 2

    // Query 2-nearest neighbors by cosine similarity
    float query[4] = {0.85f, 0.15f, 0.0f, 0.0f};
    sepal_hit_t hits[2];
    size_t n = sepal_search(vs, query, 4, 2, 0.0f, 0, hits);

    for (size_t i = 0; i < n; i++) {
        printf("Hit %zu: ref=%u score=%.4f\n", i, hits[i].ref, hits[i].score);
    }

    sepal_close(vs);
    return 0;
}
```

Compile and link:
```sh
cc semantic_search.c $(pkg-config --cflags --libs libsepal) -o semantic_search
```

👉 *For full hybrid search setups paired with `axil-auren` and `liblemma`, see [docs/SEMANTIC_SEARCH.md](../../docs/SEMANTIC_SEARCH.md).*

---

## 2. Core API Reference

The essential functions for vector storage and Approximate Nearest Neighbor (ANN) search:

| Function | Signature | Description |
| :--- | :--- | :--- |
| `sepal_open` | `sepal_vecstore_t *sepal_open(file, db)` | Open in-memory or persisted vector store |
| `sepal_put` | `int sepal_put(vs, rec_ref_t ref, vec, dim)` | Store float vector tagged to a uniform 32-bit ref |
| `sepal_search`| `size_t sepal_search(vs, q, dim, k, min_sim, m, hits)` | Run two-stage ANN search returning top-k best hits |
| `sepal_fill_approx` | `int sepal_fill_approx(vs, q, dim, m, set)` | Stream candidate pool into a recall candidate set |
| `sepal_rank` | `int sepal_rank(ctx, rec_ref_t ref, &score)` | Exact cosine reranking callback for the recall kernel |
| `rec_axis_store_typed` | `int rec_axis_store_typed(ref, data, len, qtype, ctx)` | Ingest raw float vector embeddings directly from HTOK binary containers |
| `sepal_close`| `void sepal_close(vs)` | Flush persistence and release memory |

---

## 3. Two-Stage ANN Architecture

`libsepal` converts high-dimensional float vectors into compact binary representations (`VEC1`) and searches them in two stages:

```
Query Vector ──────────────────────────────────────────────────────────────────┐
                                                                               ▼
Stage 1: Sign-Bit Sketch Prefilter (Hamming distance in L2 cache)
[ 10,000 Vector Candidates ] ──────────────► [ Top 50 Finalists (99.5% discarded) ]
                                                                               │
Stage 2: Exact Cosine Reranking (AVX2 FMA floating-point math)                 ▼
[ Top 50 Finalists ] ──────────────────────► [ Top 5 Semantic Matches (50 µs) ]
```

- **Compact Binary Footprint**: Matryoshka dimension truncation condenses 768-dimensional vectors from ~7 KB of raw text into 1,136-byte cache-friendly binary blobs (`VEC1`).
- **Bitwise Hamming Prefilter**: Stage 1 performs bitwise popcounts over sign-bit sketches in L2 CPU cache, discarding 95–99% of candidates in nanoseconds.
- **Exact AVX2 Cosine Rerank**: Stage 2 performs exact AVX2 FMA cosine reranking on only top finalists, finishing in under 50 microseconds.

---

## 4. `CORM_SEPAL_EMBED_CACHE` Persistent Cache

`libsepal` supports `CORM_SEPAL_EMBED_CACHE`, a persistent file-backed vector cache:
- **Zero Re-Embedding**: Caches text hashes and corresponding float vectors to disk, eliminating redundant neural network evaluations across application restarts.
- **Configuration**: Set `CORM_SEPAL_EMBED_CACHE=/path/to/embed.db` in your environment.

---

## 5. Installation & Build

### Prebuilt Packages
Prebuilt packages for Linux (APT / Alpine / Arch / Fedora-RHEL), macOS (Homebrew), Windows (winget / MSYS2), and OpenBSD are distributed via [tty.pt packages](https://github.com/tty-pt/ci/blob/main/docs/install.md). Use **libsepal** as the package name.

### Build from Source
```sh
make                  # builds lib/libsepal.so
make test             # run unit, property, and stress tests
sudo make install     # install headers and library to system
```

## License

BSD 2-Clause License. Copyright (c) 2025, tty-pt. See `LICENSE`.
