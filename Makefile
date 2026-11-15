all := libsepal

LDLIBS-libsepal := -lstele -lcorm -lxxhash -lm

CFLAGS := -g
CFLAGS += -O3

CFLAGS-x86_64 := -mpopcnt -mavx2 -mfma
CFLAGS-amd64 := -mpopcnt -mavx2 -mfma

WASM_SRCS := npm/wasm/sepal_wasm.c

include ../mk/include.mk

test: all
	$(MAKE) -C tests test

bench: all
	$(MAKE) -C tests bench

# tests/ has its own Makefile whose binaries are NOT covered by mk's clean,
# and its build rules carry no header deps — so a header retype (u64->u32)
# leaves stale-ABI test binaries behind that fail spuriously. Recurse.
clean: clean-tests
clean-tests:
	$(MAKE) -C tests clean

# Optional Node.js & WebAssembly bindings
NPM ?= npm

npm-build:
	@if command -v $(NPM) >/dev/null 2>&1; then \
		(cd npm && $(NPM) run build); \
	else \
		echo "Skipping npm build — $(NPM) not found"; \
	fi

npm-test: npm-build
	@if command -v $(NPM) >/dev/null 2>&1; then \
		(cd npm && $(NPM) test); \
	fi
