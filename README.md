<p align="center">
  <img src="assets/logo.png" alt="libsepal logo" width="220" />
</p>

# libsepal

[![C99](https://img.shields.io/badge/C-C99-555?logo=c)](#)
[![BSD-2-Clause](https://img.shields.io/badge/License-BSD--2--Clause-blue)](#)
[![semantic-ANN](https://img.shields.io/badge/semantic-ANN-FF6B35)](#)
[![Recall Kernel Meaning Axis](https://img.shields.io/badge/axis-meaning-blueviolet)](#)

> Ultra-fast semantic vector store with two-stage ANN search (Hamming sign-bit sketch prefilter + exact cosine rerank) in pure C.

---

## Why libsepal? (Searching by Meaning Instead of Exact Keywords)

Keyword search fails when users express ideas differently:
> *"A user searches for 'fast zero-copy RAM storage', but the documentation writes 'in-memory cache line mapping'."*

Exact keyword matching misses the document entirely. Neural embeddings solve this by converting text into high-dimensional float vectors, but computing exact cosine distances across thousands of vectors in Python or bloated vector databases is slow and memory-intensive.

`libsepal` provides a lightweight, two-stage Approximate Nearest Neighbor (ANN) vector store in pure C:

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
- **Two-Stage Filtering**: Stage 1 performs bitwise popcounts over sign-bit sketches in L2 CPU cache, discarding 95–99% of candidates in nanoseconds; Stage 2 performs exact AVX2 FMA cosine reranking on only top finalists.

---

## Recall Kernel (Meaning Axis)

`libsepal` serves as the canonical **Meaning Axis** for vector similarity and recall composition:
- **Storage Engine**: Optional file persistence and persistent vector caches backed by [libcorm](https://github.com/tty-pt/libcorm).
- **Cross-Axis Recall Kernel**: Plugs into the unified recall loop (`sepal_fill_approx` + `sepal_rank`) to score candidate sets filtered by sibling axes: [libislet](https://github.com/tty-pt/libislet) (Space), [libjoint](https://github.com/tty-pt/libjoint) (Time), and [liblemma](https://github.com/tty-pt/liblemma) (Text).

---

## End-to-End Pipeline: Pairing with `libauren`

Pair `libsepal` with [libauren](https://github.com/tty-pt/auren) for local neural semantic search:
1. **Embedding Generation**: `libauren` takes raw text and outputs normalized float vectors (e.g. 384-dimensional vectors from `bge-small`).
2. **Indexing**: `libsepal` encodes vectors into compact `VEC1` binary blobs, computing sign-bit sketches for instant Hamming prefiltering.
3. **Querying**: When a user searches, `libauren` embeds the query and `libsepal` performs two-stage ANN search in under 50 microseconds.

---

## `CORM_SEPAL_EMBED_CACHE` Persistent Cache

In `v1.1.0`, `libsepal` introduces `CORM_SEPAL_EMBED_CACHE`, a persistent file-backed vector cache:

- **Avoids Re-Embedding**: Caches text hashes and corresponding float vectors to disk, eliminating redundant neural network evaluations across application restarts.
- **Fast Zero-Copy Deserialization**: Stored in memory-mapped Corm tables for sub-microsecond retrieval.
- **Configurable Storage**: Point `CORM_SEPAL_EMBED_CACHE=/path/to/embed.db` in your environment or daemon configuration.

---

## Quickstart

Store 4D vector embeddings and perform nearest-neighbor semantic search in 15 lines:

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
cc semantic_search.c $(pkg-config --cflags --libs sepal)
```

---

## Core API (The Essential 20%)

| Function | Signature | Description |
| :--- | :--- | :--- |
| `sepal_open` | `sepal_vecstore_t *sepal_open(file, db)` | Open in-memory or persisted vector store |
| `sepal_put` | `int sepal_put(vs, rec_ref_t ref, vec, dim)` | Store float vector tagged to a uniform 32-bit ref |
| `sepal_search`| `size_t sepal_search(vs, q, dim, k, min_sim, m, hits)` | Run two-stage ANN search returning top-k best hits |
| `sepal_fill_approx` | `int sepal_fill_approx(vs, q, dim, m, set)` | Stream candidate pool into a recall candidate set |
| `sepal_rank` | `int sepal_rank(ctx, rec_ref_t ref, &score)` | Exact cosine reranking callback for the recall kernel |
| `rec_axis_store_typed` | `int rec_axis_store_typed(ref, data, len, qtype, ctx)` | Ingest raw float vector embeddings directly from HTOK binary containers |
| `sepal_close`| `void sepal_close(vs)` | Flush persistence and release memory |

👉 *For VEC1 binary blob layouts, AVX2 FMA vectorization benchmarks, and examples, see [examples/README.md](examples/README.md).*

---

## Design Philosophy: Two-Stage ANN

* **Compact VEC1 Blobs**: Uses matryoshka dimension truncation and sign-bit sketches to condense 768-dimensional vectors from ~7 KB of raw text into 1,136-byte cache-friendly binary blobs.
* **Two-Stage Filtering**: Stage 1 performs bitwise Hamming popcounts over sign sketches in L2 CPU cache (~3× faster prefilter); Stage 2 performs exact AVX2 FMA cosine reranking on only top candidates.
* **Recall Integrity**: Approximations are explicitly tracked with an owed recall bound (`m/n`), ensuring downstream ranking loops stay honest.

---

## Installation & Build

### Prebuilt Packages
Prebuilt packages are distributed on tty.pt for Linux (APT / Alpine / Arch / Fedora-RHEL), macOS (Homebrew), Windows (winget / MSYS2), and OpenBSD. Follow the [installation instructions](https://github.com/tty-pt/ci/blob/main/docs/install.md) and use **libsepal** as the package name.

### Build from Source
```sh
make                  # builds lib/libsepal.so
make test             # run unit, property, and stress tests
sudo make install     # install headers and library to system
```

## License

BSD 2-Clause License. Copyright (c) 2025, tty-pt. See `LICENSE`.
