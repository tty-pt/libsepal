# Changelog

## [0.3.0] - 2026-09-15
- SIMD/opt tier (SEPAL-PERF, post-collapse): T0 compiler flags applied for
  real + hand-written AVX2 lanes in both hot paths; besides `-O3` this is
  the first release where POPCNT is actually emitted for the prefilter
- T0 landed: `CFLAGS += -O3 -mpopcnt -mavx2 -mfma` (root `Makefile`);
  GCC auto-vectorizes the fused-cosine stage-2 loops (packed AVX2/FMA)
- AVX2 sketch: shared `sepal_sketch_words` helper (used by `sepal_sketch`
  and `sepal_blob_put`), eight `vmovmskps` per 64-float word; tail word
  (<64 dims) scalar, no OOB on non-multiple-of-64 dims (ASan-proven)
- AVX2 prefilter: `sepal_popcnt_range` (4×u64 XOR→nibble-LUT in 256-bit
  lanes) in both `prefilter` paths, scalar tail; safe on the last flat row
- Scalar `#else` fallback retained behind `#ifdef __AVX2__`
- Medians-of-3 @N=10k vs the pre-SIMD shipped config (bisect D):
  @384 pref 785→235 (3.3×), rerank 307→213 (1.44×), total 934→320 (2.9×);
  @768 pref 1233→279 (4.4×), rerank 343→210 (1.63×), total 1589→372 (4.3×)
- Recall unchanged (bit-identical semantics): mean r@10 0.366 @384,
  0.164 @768 — same as every prior tier
- Gates: full `make test` exit 0, valgrind + asan + ubsan (intrinsics
  instrumented via the T0 CFLAGS override) + fuzz clean

## [0.2.0] - 2026-09-14
- Perf tiers T0+T1+T2 (SEPAL-PERF quest, TDD RED→GREEN per tier, D1–D5)
- T0: `CFLAGS := -g -O3 -mpopcnt`; `tests/TESTING.md` documents the
  like-for-like bench + sanitizer CFLAGS-override recipe
- T1: frozen-norm rerank — stage-2 scores come from one fused dot pass
  over the blob floats with the put-frozen header norm and a
  once-per-query prefix-norm table (`sepal_search`); single fused
  dot+prefix pass in `sepal_rank`. Public `sepal_cosine` unchanged;
  score-equivalence guards (1e-6) in `test_rank.c` / `test_search.c`
- T2: flat in-memory sketch index (`idx_ref`/`idx_sk`/`idx_w` +
  XXH32 ref→row slot map), rebuilt at open, maintained on put/del,
  `prefilter()` scans contiguous rows; new public
  `sepal_index_validate()`; mutation + search-parity groups in
  `test_vecstore.c`
- Bisect harness (`SEPAL_PERF_T1` guard kept; losing `SEPAL_PERF_T2`
  fallback discarded): medians-of-3 @N=10k — T2 prefilter win confirmed
  (~2.7–3.4×: 3617→1319 µs @768, 2096→616 µs @384); T1 within machine
  noise. Machine load noise dominates absolutes (2–3× run-to-run)
- Gates re-green: full `make test` exit 0 (incl. exact-order recall
  property — no D1 relaxation needed), valgrind + asan + ubsan + fuzz
  clean (two pre-existing kernel-side suppressions, unchanged)

## [0.1.0] - 2026-09-10
- Initial release: the meaning (semantic) axis for the recall kernel (R5a)
- VEC1 binary blob store (`corm_mreg`, per-blob dims, matryoshka 256
  truncation, sign-words-over-full-dim sketch, frozen norm): 1,136 B at
  768/256; garbage/truncated/wrong-magic blobs decode to 0, never misread
- Two-stage ANN `sepal_search` (Hamming top-m → exact cosine top-k,
  `m = 10×k` default, `min_sim` at rerank, best-first, ties by asc ref)
- Kernel adapters `sepal_fill_approx` (declares `REC_SET_APPROX` with owed
  bound `m/n`; additive; `m==0` → full pool) + `sepal_rank` (exact cosine
  callback)
- Requires the site-tree libcorm kernel extension (`rec_set_set_approx` /
  `rec_set_approx` / `rec_set_recall_bound` + join propagation); system
  libcorm reinstall is a follow-on
- Tests: 8 unit tiers (3,943 assertions) + persistence integration +
  recall@k≡1 property (seeds 1/42/1337 × dims 64–768 × N up to 3000) +
  blob-never-misreads property + 10k×768 stress; valgrind + asan + ubsan
  clean (two documented kernel-side suppressions, not libsepal leaks);
  bench: two-stage beats brute force 5–28× past N=1k
- Examples: basic, twostage, persistence, kernel_form
