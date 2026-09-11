# Test suite Makefile for libsepal

CC = cc
CFLAGS = -g -Wall -Wextra -Wno-unused-function -I../include -I/home/quirinpa/site/external/libqmap/include
LDFLAGS = -L../lib -L/home/quirinpa/site/external/libqmap/lib -L/usr/lib
LDLIBS = -lsepal -lqmap -lqsys -lxxhash -lm

export LD_LIBRARY_PATH := $(abspath ../lib):/home/quirinpa/site/external/libqmap/lib:$(LD_LIBRARY_PATH)

# Test directories
UNIT_DIR = unit
INTEGRATION_DIR = integration
STRESS_DIR = stress
BENCH_DIR = benchmark
FUZZ_DIR = fuzz
PROPERTY_DIR = property

# Test source files
UNIT_TESTS = $(wildcard $(UNIT_DIR)/test_*.c)
INTEGRATION_TESTS = $(wildcard $(INTEGRATION_DIR)/test_*.c)
STRESS_TESTS = $(wildcard $(STRESS_DIR)/test_*.c)
BENCH_TESTS = $(wildcard $(BENCH_DIR)/bench_*.c)
FUZZ_TESTS = $(wildcard $(FUZZ_DIR)/fuzz_*.c)
PROPERTY_TESTS = $(wildcard $(PROPERTY_DIR)/test_*.c)

# Test executables
UNIT_BINS = $(UNIT_TESTS:.c=)
INTEGRATION_BINS = $(INTEGRATION_TESTS:.c=)
STRESS_BINS = $(STRESS_TESTS:.c=)
BENCH_BINS = $(BENCH_TESTS:.c=)
FUZZ_BINS = $(FUZZ_TESTS:.c=)
PROPERTY_BINS = $(PROPERTY_TESTS:.c=)

ALL_BINS = $(UNIT_BINS) $(INTEGRATION_BINS) $(STRESS_BINS) $(BENCH_BINS) $(PROPERTY_BINS)

# Color output
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m

.PHONY: all test test-unit test-integration test-stress test-all bench clean help
.PHONY: valgrind asan ubsan tsan coverage fuzz fuzz-standalone

all: $(ALL_BINS)

# Build rules
$(UNIT_DIR)/%: $(UNIT_DIR)/%.c test_common.h
	@echo "$(GREEN)[BUILD]$(NC) $@"
	@$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

$(INTEGRATION_DIR)/%: $(INTEGRATION_DIR)/%.c test_common.h
	@echo "$(GREEN)[BUILD]$(NC) $@"
	@$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

$(STRESS_DIR)/%: $(STRESS_DIR)/%.c test_common.h
	@echo "$(GREEN)[BUILD]$(NC) $@"
	@$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

$(BENCH_DIR)/%: $(BENCH_DIR)/%.c test_common.h
	@echo "$(GREEN)[BUILD]$(NC) $@"
	@$(CC) $(CFLAGS) -O3 $< -o $@ $(LDFLAGS) $(LDLIBS)

$(PROPERTY_DIR)/%: $(PROPERTY_DIR)/%.c test_common.h
	@echo "$(GREEN)[BUILD]$(NC) $@"
	@$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

# Test runners
test-unit: $(UNIT_BINS)
	@echo "$(YELLOW)=== Running Unit Tests ===$(NC)"
	@failed=0; \
	for test in $(UNIT_BINS); do \
		if [ -f $$test ]; then \
			./$$test || failed=$$((failed + 1)); \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "$(GREEN)All unit tests passed!$(NC)"; \
	else \
		echo "$(RED)$$failed unit test(s) failed$(NC)"; \
		exit 1; \
	fi

test-integration: $(INTEGRATION_BINS)
	@echo "$(YELLOW)=== Running Integration Tests ===$(NC)"
	@failed=0; \
	for test in $(INTEGRATION_BINS); do \
		if [ -f $$test ]; then \
			./$$test || failed=$$((failed + 1)); \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "$(GREEN)All integration tests passed!$(NC)"; \
	else \
		echo "$(RED)$$failed integration test(s) failed$(NC)"; \
		exit 1; \
	fi

test-stress: $(STRESS_BINS)
	@echo "$(YELLOW)=== Running Stress Tests ===$(NC)"
	@failed=0; \
	for test in $(STRESS_BINS); do \
		if [ -f $$test ]; then \
			./$$test || failed=$$((failed + 1)); \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "$(GREEN)All stress tests passed!$(NC)"; \
	else \
		echo "$(RED)$$failed stress test(s) failed$(NC)"; \
		exit 1; \
	fi

test-property: $(PROPERTY_BINS)
	@echo "$(YELLOW)=== Running Property-Based Tests ===$(NC)"
	@failed=0; \
	for test in $(PROPERTY_BINS); do \
		if [ -f $$test ]; then \
			./$$test || failed=$$((failed + 1)); \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "$(GREEN)All property tests passed!$(NC)"; \
	else \
		echo "$(RED)$$failed property test(s) failed$(NC)"; \
		exit 1; \
	fi

test-all: test-unit test-integration test-property test-stress
	@echo "$(GREEN)=== All Tests Passed ===$(NC)"

# Fuzz testing targets
fuzz-standalone: $(FUZZ_BINS)
	@echo "$(YELLOW)=== Running Fuzz Tests (Standalone) ===$(NC)"
	@for fuzz in $(FUZZ_BINS); do \
		if [ -f $$fuzz ]; then \
			echo "$(GREEN)[RUN]$$fuzz$(NC)"; \
			./$$fuzz || exit 1; \
		fi; \
	done

$(FUZZ_DIR)/%: $(FUZZ_DIR)/%.c
	@echo "$(GREEN)[BUILD]$$@ (standalone)$(NC)"
	@$(CC) $(CFLAGS) -DSTANDALONE $< -o $@ $(LDFLAGS) $(LDLIBS)

fuzz: $(FUZZ_BINS)
	@echo "$(YELLOW)=== Fuzz targets built ===$(NC)"
	@echo "To run fuzzers in standalone mode: make fuzz-standalone"
	@echo "To build with libFuzzer: add -DLIBFUZZER to CFLAGS and link with libFuzzer"

test: test-unit test-integration test-property test-stress
	@echo "$(GREEN)=== Core Tests Passed ===$(NC)"

bench: $(BENCH_BINS)
	@echo "$(YELLOW)=== Running Benchmarks ===$(NC)"
	@for bench in $(BENCH_BINS); do \
		if [ -f $$bench ]; then \
			./$$bench; \
		fi; \
	done

# Sanitizer builds
asan: CFLAGS += -fsanitize=address -fno-omit-frame-pointer
asan: LDFLAGS += -fsanitize=address
asan: clean all
	@echo "$(GREEN)Built with AddressSanitizer$(NC)"

ubsan: CFLAGS += -fsanitize=undefined
ubsan: LDFLAGS += -fsanitize=undefined
ubsan: clean all
	@echo "$(GREEN)Built with UndefinedBehaviorSanitizer$(NC)"

tsan: CFLAGS += -fsanitize=thread
tsan: LDFLAGS += -fsanitize=thread
tsan: clean all
	@echo "$(GREEN)Built with ThreadSanitizer$(NC)"

# Memory checking
valgrind: all
	@echo "$(YELLOW)=== Running tests with Valgrind ===$(NC)"
	@for test in $(UNIT_BINS) $(INTEGRATION_BINS); do \
		if [ -f $$test ]; then \
			echo "$(YELLOW)[VALGRIND]$(NC) $$test"; \
			valgrind --leak-check=full --error-exitcode=1 --quiet --suppressions=valgrind.supp ./$$test || exit 1; \
		fi; \
	done
	@echo "$(GREEN)Valgrind checks passed!$(NC)"

# Code coverage
coverage: CFLAGS += --coverage
coverage: LDFLAGS += --coverage
coverage: clean all test
	@echo "$(YELLOW)=== Generating Coverage Report ===$(NC)"
	@lcov --capture --directory . --output-file coverage.info
	@lcov --remove coverage.info '/usr/*' --output-file coverage.info
	@genhtml coverage.info --output-directory coverage_html
	@echo "$(GREEN)Coverage report generated in coverage_html/$(NC)"

# Cleanup
clean:
	@echo "$(YELLOW)[CLEAN]$(NC) Removing test binaries and artifacts"
	@rm -f $(ALL_BINS) $(FUZZ_BINS)
	@rm -f *.gcda *.gcno *.gcov coverage.info
	@rm -rf coverage_html
	@rm -f /tmp/test_sepal_*.db

# Help
help:
	@echo "$(GREEN)libsepal Test Suite$(NC)"
	@echo ""
	@echo "Targets:"
	@echo "  $(YELLOW)test$(NC)              - Run unit + integration + property tests"
	@echo "  $(YELLOW)test-unit$(NC)         - Run only unit tests"
	@echo "  $(YELLOW)test-integration$(NC)  - Run only integration tests"
	@echo "  $(YELLOW)test-stress$(NC)       - Run stress tests"
	@echo "  $(YELLOW)test-property$(NC)     - Run property-based tests"
	@echo "  $(YELLOW)test-all$(NC)          - Run all tests (including stress)"
	@echo "  $(YELLOW)bench$(NC)             - Run performance benchmarks"
	@echo "  $(YELLOW)fuzz$(NC)              - Build fuzz test targets"
	@echo "  $(YELLOW)fuzz-standalone$(NC)   - Build and run fuzz tests in standalone mode"
	@echo "  $(YELLOW)valgrind$(NC)          - Run tests with Valgrind"
	@echo "  $(YELLOW)asan$(NC)              - Build with AddressSanitizer"
	@echo "  $(YELLOW)ubsan$(NC)             - Build with UndefinedBehaviorSanitizer"
	@echo "  $(YELLOW)tsan$(NC)              - Build with ThreadSanitizer"
	@echo "  $(YELLOW)coverage$(NC)          - Generate code coverage report"
	@echo "  $(YELLOW)clean$(NC)             - Remove all test binaries"
	@echo "  $(YELLOW)help$(NC)              - Show this help message"
