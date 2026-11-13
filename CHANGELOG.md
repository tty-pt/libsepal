# Changelog

## [1.4.1] - 2026-11-13

- **Disjoint Global-Base & Detached Buffer Safety (`sepal.wasm`)**:
  - Rebuilt `sepal.wasm` shared-memory reactor at `--global-base=1572864` and fixed memory growth ordering before TypedArray construction.
  - Upgraded `@tty-pt/corm` to `>=2.3.2` (`^2.3.2`) and `@tty-pt/stele` to `>=1.6.1` (`^1.6.1`).

## [1.4.0] - 2026-11-13

- **Substrate & Corm v2.3.1 Alignment**:
  - Upgraded dependencies to @tty-pt/corm@^2.3.1 (>=2.3.1) and @tty-pt/stele@^1.6.0 (>=1.6.0).
  - Maintained zero-copy vector candidate ranking and Matryoshka dimension scaling.

## [1.3.2] - 2026-11-03

- **Standalone `@tty-pt/sepal` Package & WebAssembly Vector Engine**:
  - Published `@tty-pt/sepal` providing Node-API native bindings and pure WebAssembly fallback (`sepal.wasm`).
  - Added `npm/wasm/sepal_wasm.c` exporting `sepal_cosine`, `sepal_wasm_sketch`, and `sepal_wasm_hamming` via `ISO_EXPORT`.
  - Links with `-Wl,--import-memory` for zero-copy vector candidate ranking alongside `@tty-pt/corm` and `@tty-pt/joint`.
  - Adopted strict `peerDependencies` (`@tty-pt/stele: ">=1.5.0"`, `@tty-pt/corm: ">=2.2.0"`) and `devDependencies` (`@tty-pt/stele: "^1.5.2"`, `@tty-pt/corm: "^2.2.4"`).
  - Migrated build to centralized `make wasm` in `external/mk/include.mk` with post-link `wasm-opt -O3` optimization.
- **HTOK Vector Embedding Ingestion (`rec_axis_store_typed`)**:
  - Exported `rec_axis_store_typed` enabling direct ingestion of vector embeddings from `libhyle-source`'s `.tok` binary containers into the Sepal vector store.

## [1.1.0]

- **Vector Embedding Cache (`CORM_SEPAL_EMBED_CACHE`)**: File-backed disk cache for vector embeddings to avoid redundant network/model fetch overhead.
- **Cormr Recall Kernel Integration**: Updated headers and tests to consume `<ttypt/corm.h>` and `<ttypt/cormr.h>`.
- **Lemma Text Indexing Alignment**: Updated ecosystem references and test fixtures from `libstoma` to `liblemma`.
- **Cleaned Build Flags**: Removed unused compiler options.

## [1.0.0]

- **First stable release** — semantic (meaning) axis vector store for the recall kernel: store float vectors with per-blob dimensions and query by meaning.
- **Dim-tagged VEC1 blobs**: each vector stores its own exact-stage dim (`min(full_dim, 256)` matryoshka prefix) and a full-dim sign sketch — 1,136 bytes at 768/256 instead of ~7 KB of text. `sepal_blob_len`/`sepal_blob_hdr`/`sepal_blob_put` expose the format; garbage/truncation always decodes to 0, never a misread.
- **Two-stage ANN search**: Hamming popcount prefilter over the sign sketch picks the top-`m` candidates (`m` = 10×`k` by default, capped at N), then an exact cosine rerank keeps the best-`k`; `min_sim` filter at rerank. `sepal_search` returns hits best-first with ties by ascending ref.
- **Recall-kernel adapters**: `sepal_fill_approx` streams top-`m` refs into a sealed `rec_set_t` marked `REC_SET_APPROX` with an owed recall bound (`m/n`, propagated by the kernel extension); `sepal_rank` (a `rec_score_fn`) scores one ref exactly. Phase 2A store half lands too — `rec_axis_store`/`rec_axis_unstore`/`rec_axis_readback` (store is keyed by ref, replace-in-place; `unstore` normalizes absent → 0) and `rec_axis_open` (file path spec, or empty → memory-only store).
- **File persistence**: optional libcorm sidecar; `sepal_close` saves (never closes — the file would truncate to 0), the vector count is restored on open. `sepal_index_validate` checks internal consistency.
- **Performance**: flat sketch index (L2-resident at N=10k, ~3× faster prefilter), frozen-norm fused-dot rerank (semantically `sepal_cosine` within 1e-6), AVX2 fast paths (`vmovmskps` sign bits, XOR→nibble-LUT popcount, FMA cosine) with a scalar fallback. Vs the pre-AVX2 shipped config: total search 2.9×/4.3× faster at 384/768 dim.
- **API**: `sepal_open`/`sepal_close`/`sepal_put`/`sepal_del`/`sepal_get`/`sepal_dim`/`sepal_full_dim`/`sepal_n`, `sepal_cosine`, `sepal_sketch`, `sepal_search`, `sepal_configure_embeddings` (set the embedding dim at runtime), plus the kernel adapters. Caller-opaque `rec_ref_t` (u32) refs — the store never maps refs to schemas.
- Depends on `libcorm` for the recall-kernel approximation flag.
