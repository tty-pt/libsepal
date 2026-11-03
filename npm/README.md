<p align="center">
  <img src="https://raw.githubusercontent.com/tty-pt/libsepal/main/assets/logo.png" alt="@tty-pt/sepal logo" width="220" />
</p>

# @tty-pt/sepal

[![npm version](https://img.shields.io/badge/npm-v0.1.0-CB3837?logo=npm)](#)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD--2--Clause-blue)](#)
[![TypeScript Ready](https://img.shields.io/badge/TypeScript-Ready-3178C6?logo=typescript)](#)
[![Node.js Native](https://img.shields.io/badge/Node.js-N--API-339933?logo=node.js)](#)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-Reactor-654FF0?logo=webassembly)](#)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0-success)](#)

> Universal Node.js and Browser bindings for **`libsepal`**, the pure-C dense vector similarity, bitwise Hamming sketch, and two-stage ANN search engine. Represents the canonical **Meaning / Semantic Axis** in the Corm Recall Kernel.

---

## 1. Quickstart

```bash
npm install @tty-pt/sepal
```

```typescript
import { cosineSimilarity, VectorIndex } from "@tty-pt/sepal";

// 1. Normalized vector cosine similarity calculation
const sim = cosineSimilarity([0.1, 0.9, 0.0], [0.0, 1.0, 0.0]);
console.log("Similarity:", sim); // ~0.99

// 2. Vector indexing & two-stage ANN search
const index = new VectorIndex("./embeddings.sepal");
index.put(101, [0.12, 0.85, 0.44]);
index.put(102, [0.90, 0.10, 0.05]);

// Query top-5 nearest candidates above 0.5 minimum similarity:
const hits = index.search([0.10, 0.80, 0.40], 5, 0.5);
console.log("Top matches:", hits); // [{ ref: 101, score: 0.99 }, ...]
```

---

## 2. Practical Use Cases

- **AI Agent Memory**: Used by local agents (`pi-mm`, `axil-auren`) for sub-millisecond semantic retrieval across 100k+ embedding vectors.
- **Client-Side Semantic Autocomplete**: Rank product, document, or candidate embeddings in client tabs without making backend roundtrips.
- **Deduplication & Clustering**: Fast cosine matching with bitwise sign-bit sketch pre-filtering.

---

## 3. JavaScript / TypeScript API Reference

### `cosineSimilarity(vecA: number[] | Float32Array, vecB: number[] | Float32Array): number`
Computes the normalized cosine similarity $\frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$ between two dense float vectors. **Executes strictly in compiled WebAssembly / C** via `sepal_cosine()`.

### `VectorIndex`
Persistent or in-memory vector index:
* `constructor(filePath?: string)`: Opens database or creates in-memory vector store.
* `.put(ref: number, vector: number[]): boolean`: Stores dense embedding vector for reference ID.
* `.search(queryVector: number[], k?: number, minScore?: number): Array<{ ref: number, score: number }>`: Performs cosine nearest-neighbor search, returning top-k matching references ranked by similarity.

### `SepalVectorAxis`
Plug-and-play Semantic Vector Axis adapter for the Corm Recall Kernel (`@tty-pt/corm`):
* `.registerVector(ref: number, vector: number[]): void`
* `.rank(params: { queryVector: number[] }, ref: number): number`

### `getSepalWasm(): WebAssembly.Instance`
Returns the WebAssembly instance loaded on the Stele shared-memory substrate.

---

## 4. Dual-Engine Runtime

- **Node.js**: Native C99 addon (`build/Release/sepal_node.node`) with hardware AVX2/popcount SIMD acceleration.
- **Browser & Edge**: Precompiled `dist/sepal.wasm` reactor with zero native compilation prerequisites.

---

## 4. C Development Headers via NPM

```makefile
CFLAGS += -I./node_modules/@tty-pt/sepal/include
LDLIBS += -lsepal
```

---

## 5. Building from Source

```bash
make npm-build    # Compiles native addon and WASM reactor
make npm-test     # Runs verification test suite
```

## License

BSD 2-Clause License. Copyright (c) 2026, tty-pt.
