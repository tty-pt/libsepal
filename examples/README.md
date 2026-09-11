# libsepal examples

Build: `make -C ..` first (builds `lib/libsepal.so`), then `make` here.
Run: `make run` (needs the `LD_LIBRARY_PATH` the Makefile exports: the
repo's own `lib/` plus the site-tree `libqmap` carrying the kernel
approximation flag).

| Example        | Shows                                                        |
|----------------|--------------------------------------------------------------|
| `basic`        | open → put → get → search (the 30-second tour)               |
| `twostage`     | `sepal_fill_approx` + `sepal_rank` spelled out, then the equivalent `sepal_search` |
| `persistence`  | file-backed store across runs (close → reopen, count restored) |
| `kernel_form`  | fill ∩ exact set → join keeps the approx flag → exact rerank |
