# Changelog

## [0.1.0] - 2026-09-10
- Initial release: the meaning (semantic) axis for the recall kernel (R5a)
- VEC1 binary blob store (`qmap_mreg`, per-blob dims, matryoshka 256
  truncation, sign-words-over-full-dim sketch, frozen norm): 1,136 B at
  768/256; garbage/truncated/wrong-magic blobs decode to 0, never misread
- Two-stage ANN `sepal_search` (Hamming top-m → exact cosine top-k,
  `m = 10×k` default, `min_sim` at rerank, best-first, ties by asc ref)
- Kernel adapters `sepal_fill_approx` (declares `REC_SET_APPROX` with owed
  bound `m/n`; additive; `m==0` → full pool) + `sepal_rank` (exact cosine
  callback)
- Requires the site-tree libqmap kernel extension (`rec_set_set_approx` /
  `rec_set_approx` / `rec_set_recall_bound` + join propagation); system
  libqmap reinstall is a follow-on
- Tests: 8 unit tiers (3,943 assertions) + persistence integration +
  recall@k≡1 property (seeds 1/42/1337 × dims 64–768 × N up to 3000) +
  blob-never-misreads property + 10k×768 stress; valgrind + asan + ubsan
  clean (two documented kernel-side suppressions, not libsepal leaks);
  bench: two-stage beats brute force 5–28× past N=1k
- Examples: basic, twostage, persistence, kernel_form
