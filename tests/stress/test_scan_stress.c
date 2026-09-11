/* test_scan_stress.c — N=10k × 768-dim store: full scan + two-stage search
 * smoke. Guards against O(N^2)/alloc-churn regressions. */

#include "../test_common.h"

#define BIG_N    10000
#define BIG_DIM  768
#define CHK_CAP  4096

static void
test_big_scan_and_search(void)
{
	printf("=== stress: N=%d × %d-dim put + two-stage search ===\n",
	       BIG_N, BIG_DIM);

	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	sepal_rng_t r = { 81 };
	float *v = malloc(BIG_DIM * sizeof(float));
	ASSERT_NOT_NULL(v);
	for (size_t i = 0; i < BIG_N; i++) {
		rng_unit_vector(&r, v, BIG_DIM);
		ASSERT_EQ(sepal_put(vs, (rec_ref_t)i, v, BIG_DIM), 0);
	}
	ASSERT_EQ(sepal_n(vs), BIG_N);

	float q[BIG_DIM];
	sepal_hit_t hits[16];
	size_t total = 0;
	for (size_t qi = 0; qi < 25; qi++) {
		rng_unit_vector(&r, q, BIG_DIM);
		size_t h = sepal_search(vs, q, BIG_DIM, 10, 0.1f, 100, hits);
		ASSERT(h <= 10, "bounded by k");
		total += h;
		for (size_t i = 0; i < h; i++) {
			ASSERT(hits[i].score + 1e-6f >= 0.1f, "above min_sim");
			if (i > 0)
				ASSERT(hits[i - 1].score >= hits[i].score, "non-increasing");
		}
	}
	ASSERT(total > 0, "hits produced");

	/* approximate fill over the full store must contain everything */
	rec_set_t *s = rec_set_new();
	ASSERT_NOT_NULL(s);
	ASSERT_EQ(sepal_fill_approx(vs, q, BIG_DIM, BIG_N, -1.0f, s), 0);
	ASSERT_EQ(rec_set_count(s), BIG_N);
	ASSERT(rec_set_approx(s) == REC_SET_APPROX, "approx declared");
	rec_set_free(s);

	/* del a slice and re-check counts */
	for (size_t i = 0; i < 100; i++)
		ASSERT_EQ(sepal_del(vs, (rec_ref_t)i), 0);
	ASSERT_EQ(sepal_n(vs), BIG_N - 100);

	sepal_close(vs);
	free(v);
}

int
main(void)
{
	test_big_scan_and_search();
	return test_summary();
}