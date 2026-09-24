/* test_fill.c — sepal_fill_approx: kernel-form approximate fill (D3/D10). */

#include "../test_common.h"

static sepal_vecstore_t *
build_corpus(size_t n, uint64_t seed, size_t dim)
{
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	if (!vs)
		return NULL;
	float *v = malloc(dim * sizeof(float));
	if (!v) {
		sepal_close(vs);
		return NULL;
	}
	sepal_rng_t r = { seed };
	for (size_t i = 0; i < n; i++) {
		rng_unit_vector(&r, v, dim);
		sepal_put(vs, (rec_ref_t)i, v, dim);
	}
	free(v);
	return vs;
}

static void
test_fill_basic(void)
{
	printf("=== fill: approximate declared, bound heuristic ===\n");
	sepal_vecstore_t *vs = build_corpus(100, 41, 64);
	ASSERT_NOT_NULL(vs);

	float q[64];
	sepal_rng_t r = { 42 };
	rng_unit_vector(&r, q, 64);

	rec_set_t *s = rec_set_new();
	ASSERT_NOT_NULL(s);
	ASSERT_EQ(sepal_fill_approx(vs, q, 64, 100, 0.0f, s), 0);
	ASSERT(rec_set_approx(s) == REC_SET_APPROX, "fill declares approximate");
	ASSERT_NEAR(rec_set_recall_bound(s), 1.0f, 1e-6f);       /* m >= n */
	ASSERT_EQ(rec_set_count(s), 100);                        /* m == n full */

	rec_set_t *s2 = rec_set_new();
	ASSERT_EQ(sepal_fill_approx(vs, q, 64, 10, 0.0f, s2), 0);
	ASSERT(rec_set_approx(s2) == REC_SET_APPROX, "bound in (0,1]");
	ASSERT_NEAR(rec_set_recall_bound(s2), 0.1f, 1e-4f);
	ASSERT_EQ(rec_set_count(s2), 10);
	rec_set_free(s2);
	rec_set_free(s);
	sepal_close(vs);
}

static void
test_fill_clamp(void)
{
	printf("=== fill: m clamped to store size (0 and >n) ===\n");
	sepal_vecstore_t *vs = build_corpus(3, 41, 8);
	ASSERT_NOT_NULL(vs);

	float q[8];
	sepal_rng_t r = { 42 };
	rng_unit_vector(&r, q, 8);

	rec_set_t *s1 = rec_set_new();
	ASSERT_NOT_NULL(s1);
	ASSERT_EQ(sepal_fill_approx(vs, q, 8, 999, 0.0f, s1), 0);
	ASSERT(rec_set_count(s1) == 3, "m > n clamps to n");
	ASSERT_NEAR(rec_set_recall_bound(s1), 1.0f, 1e-6f);
	rec_set_free(s1);

	rec_set_t *s0 = rec_set_new();
	ASSERT_NOT_NULL(s0);
	ASSERT_EQ(sepal_fill_approx(vs, q, 8, 0, 0.0f, s0), 0);
	ASSERT(rec_set_count(s0) == 3, "m == 0 defaults to n");
	rec_set_free(s0);
	sepal_close(vs);
}

static void
test_fill_contains_topk(void)
{
	printf("=== fill: with full pool, top-k of brute force ⊆ fill set ===\n");
	sepal_vecstore_t *vs = build_corpus(200, 43, 256);
	ASSERT_NOT_NULL(vs);

	float *base = malloc(200 * 256 * sizeof(float));
	ASSERT_NOT_NULL(base);
	for (size_t i = 0; i < 200; i++)
		sepal_get(vs, (rec_ref_t)i, base + i * 256, 256);

	sepal_rng_t r = { 44 };
	float q[256];
	sepal_bf_hit_t bf[8];

	for (size_t qi = 0; qi < 40; qi++) {
		rng_unit_vector(&r, q, 256);
		size_t bfn = brute_force_search(base, 200, 256, 256,
		                                q, 256, 5, 0.05f, bf);
		rec_set_t *s = rec_set_new();
		ASSERT_EQ(sepal_fill_approx(vs, q, 256, 200, 0.05f, s), 0);

		const rec_ref_t *got = rec_set_at(s);
		for (size_t i = 0; i < bfn; i++) {
			int found = 0;
			for (size_t j = 0; j < rec_set_count(s); j++)
				if (got[j] == bf[i].ref)
					found = 1;
			ASSERT(found, "top-k ball is inside the filled set");
			if (!found)
				break;
		}
		rec_set_free(s);
	}
	free(base);
	sepal_close(vs);
}

static void
test_fill_additive(void)
{
	printf("=== fill: additive over an existing, sealed set ===\n");
	sepal_vecstore_t *vs = build_corpus(60, 45, 32);
	ASSERT_NOT_NULL(vs);

	float q[32];
	sepal_rng_t r = { 46 };
	rng_unit_vector(&r, q, 32);

	rec_set_t *s = rec_set_new();
	rec_set_push(s, 999);
	rec_set_seal(s);
	ASSERT(rec_set_approx(s) == REC_SET_EXACT, "pre-fill set is exact");

	ASSERT_EQ(sepal_fill_approx(vs, q, 32, 5, 0.0f, s), 0);
	ASSERT(rec_set_approx(s) == REC_SET_APPROX, "declared approximate");
	ASSERT_EQ(rec_set_count(s), 6);                        /* 999 + top 5 */

	const rec_ref_t *got = rec_set_at(s);
	int have_999 = 0;
	for (size_t i = 0; i < rec_set_count(s); i++)
		if (got[i] == 999)
			have_999 = 1;
	ASSERT(have_999, "pre-existing ref preserved");
	rec_set_free(s);
	sepal_close(vs);
}

static void
test_fill_intersect_propagation(void)
{
	printf("=== fill: ∩ with an exact set → approx + bound = min ===\n");
	sepal_vecstore_t *vs = build_corpus(100, 47, 64);
	ASSERT_NOT_NULL(vs);

	float q[64];
	sepal_rng_t r = { 48 };
	rng_unit_vector(&r, q, 64);

	rec_set_t *approx = rec_set_new();
	ASSERT_EQ(sepal_fill_approx(vs, q, 64, 10, 0.0f, approx), 0);

	rec_set_t *exact = rec_set_new();
	rec_set_push(exact, 0);
	rec_set_push(exact, 1);
	rec_set_push(exact, 2);
	rec_set_seal(exact);

	rec_set_t *joined = rec_set_new();
	ASSERT_EQ(rec_set_intersect(joined, exact, approx), 0);
	ASSERT(rec_set_approx(joined) == REC_SET_APPROX, "exact∩approx → approx");
	ASSERT_NEAR(rec_set_recall_bound(joined), 0.1f, 1e-4f);   /* min(1.0, 0.1) */

	rec_set_t *sub = rec_set_new();
	ASSERT_EQ(rec_set_subtract(sub, approx, exact), 0);
	ASSERT(rec_set_approx(sub) == REC_SET_APPROX, "approx−exact keeps approx");

	rec_set_free(approx);
	rec_set_free(exact);
	rec_set_free(joined);
	rec_set_free(sub);
	sepal_close(vs);
}

static void
test_fill_guards(void)
{
	printf("=== fill: NULL args / bad dims → -1 ===\n");
	sepal_vecstore_t *vs = build_corpus(10, 49, 32);
	ASSERT_NOT_NULL(vs);

	float q[32];
	sepal_rng_t r = { 50 };
	rng_unit_vector(&r, q, 32);
	rec_set_t *s = rec_set_new();

	ASSERT_EQ(sepal_fill_approx(NULL, q, 32, 5, 0.0f, s), -1);
	ASSERT_EQ(sepal_fill_approx(vs, NULL, 32, 5, 0.0f, s), -1);
	ASSERT_EQ(sepal_fill_approx(vs, q, 0, 5, 0.0f, s), -1);
	ASSERT_EQ(sepal_fill_approx(vs, q, SEPAL_VEC_MAX + 1, 5, 0.0f, s), -1);
	ASSERT_EQ(sepal_fill_approx(vs, q, 32, 5, 0.0f, NULL), -1);

	ASSERT(rec_set_approx(s) == REC_SET_EXACT, "guards leave set unchanged");
	rec_set_free(s);
	sepal_close(vs);
}

int
main(void)
{
	test_fill_basic();
	test_fill_clamp();
	test_fill_contains_topk();
	test_fill_additive();
	test_fill_intersect_propagation();
	test_fill_guards();
	return test_summary();
}