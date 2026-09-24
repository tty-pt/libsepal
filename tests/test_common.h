/*
 * test_common.h — shared utilities for the libsepal test suite.
 *
 * libgeo-style: assertion macros with colored output, per-file counters,
 * immediate exit on the first failure. Extras: a deterministic fixed-seed
 * LCG (no libc rand), vector/unit-vector builders, and an exact brute-force
 * reference search (scan + cosine) used to validate the two-stage ANN.
 */

#ifndef SEPAL_TEST_COMMON_H
#define SEPAL_TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <ttypt/sepal.h>

#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BOLD    "\033[1m"

static int test_count = 0;
static int test_failed = 0;
#define CHECK_UNUSED(name) ((void)name)

static void test_fail(const char *file, int line, const char *expr) {
	fprintf(stderr, "%s    FAIL:%s %s:%d: Assertion failed: %s\n",
	        COLOR_RED, COLOR_RESET, file, line, expr);
	test_failed++;
	return;
}

#define ASSERT(expr, ...) do { \
		test_count++; \
		if (!(expr)) test_fail(__FILE__, __LINE__, #expr); \
	} while (0)

#define ASSERT_EQ(a, b) do { \
		test_count++; \
		long long _a = (long long)(a), _b = (long long)(b); \
		if (_a != _b) { \
			fprintf(stderr, "%s    %s:%d: %s == %s  (%lld != %lld)%s\n", \
			        COLOR_YELLOW, __FILE__, __LINE__, #a, #b, _a, _b, COLOR_RESET); \
			test_fail(__FILE__, __LINE__, #a " == " #b); \
		} \
	} while (0)

#define ASSERT_NE(a, b) do { \
		test_count++; \
		long long _a = (long long)(a), _b = (long long)(b); \
		if (_a == _b) { \
			fprintf(stderr, "%s    %s:%d: %s != %s (both %lld)%s\n", \
			        COLOR_YELLOW, __FILE__, __LINE__, #a, #b, _a, COLOR_RESET); \
			test_fail(__FILE__, __LINE__, #a " != " #b); \
		} \
	} while (0)

#define ASSERT_LE(a, b) do { \
		test_count++; \
		long long _a = (long long)(a), _b = (long long)(b); \
		if (_a > _b) test_fail(__FILE__, __LINE__, #a " <= " #b); \
	} while (0)

#define ASSERT_GE(a, b) do { \
		test_count++; \
		long long _a = (long long)(a), _b = (long long)(b); \
		if (_a < _b) test_fail(__FILE__, __LINE__, #a " >= " #b); \
	} while (0)

#define ASSERT_NULL(p) ASSERT((p) == NULL)
#define ASSERT_NOT_NULL(p) ASSERT((p) != NULL)

#define ASSERT_NEAR(a, b, eps) do { \
		test_count++; \
		double _a = (double)(a), _b = (double)(b); \
		if (fabs(_a - _b) > (eps)) { \
			fprintf(stderr, "%s    %s:%d: %s ~= %s (%g vs %g)%s\n", \
			        COLOR_YELLOW, __FILE__, __LINE__, #a, #b, _a, _b, COLOR_RESET); \
			test_fail(__FILE__, __LINE__, #a " ~= " #b); \
		} \
	} while (0)

static int test_summary(void) {
	if (test_failed == 0) {
		printf("%sALL %d ASSERTIONS PASSED%s\n", COLOR_GREEN, test_count, COLOR_RESET);
		return 0;
	}
	printf("%s%d/%d ASSERTION(S) FAILED%s\n", COLOR_RED, test_failed,
	       test_count, COLOR_RESET);
	return 1;
}

/* ----------------------------------------------------------------------.
 * Deterministic PRNG (LCG; fixed seeds => reproducible suites)           */

typedef struct {
	uint64_t s;
} sepal_rng_t;

static inline uint64_t
rng_next(sepal_rng_t *r)
{
	r->s = r->s * 6364136223846793005ULL + 1442695040888963407ULL;
	return r->s;
}

static inline float
rng_unit(sepal_rng_t *r)
{
	/* uniform in [-1,1] */
	return (float)((double)(rng_next(r) & 0xFFFFFFFFu) / 0xFFFFFFFFu * 2.0 - 1.0);
}

/* ----------------------------------------------------------------------.
 * Vector builders                                                       */

/* Fill v[0..n) with ~unit magnitude, then normalize to a unit vector. */
static inline void
rng_unit_vector(sepal_rng_t *r, float *v, size_t n)
{
	double s = 0.0;
	for (size_t i = 0; i < n; i++) {
		v[i] = rng_unit(r);
		s += (double)v[i] * (double)v[i];
	}
	if (s > 0.0) {
		double inv = 1.0 / sqrt(s);
		for (size_t i = 0; i < n; i++)
			v[i] = (float)((double)v[i] * inv);
	}
}

static inline float
ref_cosine(const float *a, const float *b, size_t n)
{
	double dot = 0.0, na = 0.0, nb = 0.0;
	for (size_t i = 0; i < n; i++) {
		dot += (double)a[i] * (double)b[i];
		na += (double)a[i] * (double)a[i];
		nb += (double)b[i] * (double)b[i];
	}
	if (na <= 0.0 || nb <= 0.0)
		return 0.0f;
	return (float)(dot / sqrt(na * nb));
}

/* ----------------------------------------------------------------------.
 * Exact brute-force reference over a flat array of rows.
 *   base    = rows laid out with stride row_dim floats each
 *   n_vecs  = number of rows
 *   row_dim = full dim of a stored row (the store's stride)
 *   cap     = matryoshka exact-stage cap (SEPAL_EXACT_DIM); the stored
 *             exact dim is min(row_dim, cap), and must also be <= qdim.
 * out[] gets up to k hits >= min_sim, best-first, ties by ascending ref.  */

typedef struct {
	rec_ref_t ref;
	float     score;
} sepal_bf_hit_t;

static inline size_t
brute_force_search(const float *base, size_t n_vecs, size_t row_dim,
                   size_t cap, const float *q, size_t qdim,
                   size_t k, float min_sim, sepal_bf_hit_t *out)
{
	size_t cnt = 0;
	sepal_bf_hit_t *pool = malloc(n_vecs > 0 ? n_vecs * sizeof(*pool) : 1);
	if (!pool)
		return 0;
	size_t stored = row_dim < cap ? row_dim : cap;
	for (size_t i = 0; i < n_vecs; i++) {
		size_t dim = stored < qdim ? stored : qdim;
		if (dim == 0)
			continue;
		/* use the library cosine (float) so score bits match the store's
		 * rerank exactly — boundary/order comparisons stay deterministic */
		float s = sepal_cosine(base + i * row_dim, q, dim);
		if (s + 0.0f < min_sim)
			continue;
		pool[cnt].ref = (rec_ref_t)i;
		pool[cnt].score = s;
		cnt++;
	}
	/* insertion sort: score desc, ties ref asc */
	for (size_t i = 1; i < cnt; i++) {
		sepal_bf_hit_t h = pool[i];
		size_t j = i;
		while (j > 0 &&
		       (pool[j - 1].score < h.score ||
		        (pool[j - 1].score == h.score && pool[j - 1].ref > h.ref))) {
			pool[j] = pool[j - 1];
			j--;
		}
		pool[j] = h;
	}
	size_t n = cnt < k ? cnt : k;
	memcpy(out, pool, n * sizeof(*out));
	free(pool);
	return n;
}

#endif /* SEPAL_TEST_COMMON_H */