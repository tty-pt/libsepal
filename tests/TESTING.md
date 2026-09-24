# TESTING — libsepal test tiers

All tiers live under `tests/`. The suite Makefile auto-discovers every
`test_*.c` / `bench_*.c` / `fuzz_*.c`, so adding a file is enough.

Richness requires the lib built first: run `make` at the repo root (this
descends into `tests/` only for the `test:`/`bench:` targets). If you work
inside `tests/` directly, run `make -C ..` once to build `lib/sepal.so`.

The binaries have no rpath: the Makefile exports `LD_LIBRARY_PATH` (its own
`lib/` plus the site-tree `libcorm` that carries the kernel approximation
flag). Everything here assumes the site `external/libcorm` kernel extension
has been built (`make` in `/home/quirinpa/site/external/libcorm`).

## Tiers

| Target                | Runs                                                        |
|-----------------------|-------------------------------------------------------------|
| `make test`           | unit + integration + property + stress (default)            |
| `make test-all`       | the above (alias with a PASSED banner)                      |
| `make test-unit`      | unit tests only: blob, cosine, sketch, matryoshka, vecstore, search, fill, rank |
| `make test-integration` | `test_persistence` (file-backed reopen parity)            |
| `make test-property`  | `test_recall_prop` (recall@k≡1), `test_blob_props` (never misreads) |
| `make test-stress`    | `test_scan_stress` (N=10k × 768 scan/search smoke)          |
| `make bench`          | `bench_search` (two-stage vs brute force, records study §12.4) |
| `make fuzz` / `make fuzz-standalone` | build / run standalone fuzzers |
| `make valgrind`       | unit + integration under valgrind (leak-check, error-exitcode 1) |
| `make asan` / `ubsan` / `tsan` | rebuild + run unit tests under the sanitizer        |
| `make coverage`       | lcov coverage report into `coverage_html/`                  |

Run a single binary directly for iteration, e.g.
`./unit/test_search` (the Makefile exports `LD_LIBRARY_PATH`, a bare shell
does not — prefix it or `make test-unit`).

## Determinism

Property tests use a fixed-seed LCG (seeds 1, 42, 1337) — the corpus and
queries are identical on every run, so recall assertions are reproducible.
The brute-force reference is computed with the library's own float cosine,
so float/double boundary disagreements can never flake the comparisons.