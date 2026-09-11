/* test_recall_prop.c — recall@k ≡ 1.0 for the two-stage pipeline.
 *
 * Fixed-LCG seeds x dims x store sizes: with a full prefilter pool the
 * approximate fill must contain the exact brute-force top-k ball for every
 * query, and sepal_search must reproduce the brute-force ordering. */

#include "../test_common.h"

typedef struct {
	size_t n_store;
	size_t dim;
	size_t n_query;
} prop_case_t;

static void
run_case(const prop_case_t *pc, uint64_t seed)
{
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	if (!vs) {
		ASSERT_NOT_NULL(vs);
		return;
	}

	float *v = malloc(pc->dim * sizeof(float));
	float *q = malloc(pc->dim * sizeof(float));
	float *base = malloc(pc->n_store * pc->dim * sizeof(float));
	ASSERT_NOT_NULL(v);
	ASSERT_NOT_NULL(q);
	ASSERT_NOT_NULL(base);

	sepal_rng_t r = { seed };
	for (size_t i = 0; i < pc->n_store; i++) {
		rng_unit_vector(&r, v, pc->dim);
		ASSERT_EQ(sepal_put(vs, (rec_ref_t)i, v, pc->dim), 0);
		memcpy(base + i * pc->dim, v, pc->dim * sizeof(float));
	}

	size_t k = 10;
	float min_sim = 0.0f;
	sepal_hit_t sh[16];
	sepal_bf_hit_t bf[16];

	for (size_t qi = 0; qi < pc->n_query; qi++) {
		rng_unit_vector(&r, q, pc->dim);

		size_t shn = sepal_search(vs, q, pc->dim, k, -1.0f, pc->n_store, sh);
		size_t bfn = brute_force_search(base, pc->n_store, pc->dim, 256,
		                                q, pc->dim, k, -1.0f, bf);

		/* full pool ⇒ the two-stage result must be exactly the brute-force
		 * result, in the same order */
		ASSERT_EQ(shn, bfn);
		size_t common = shn < k ? shn : k;
		for (size_t i = 0; i < common; i++) {
			ASSERT_EQ(sh[i].ref, bf[i].ref);
			ASSERT_NEAR(sh[i].score, bf[i].score, 1e-4f);
		}

		/* approximate fill (m = n) superset of the top-k ball */
		rec_set_t *s = rec_set_new();
		if (!s) {
			ASSERT_NOT_NULL(s);
			return;
		}
		size_t m = pc->n_store;
		ASSERT_EQ(sepal_fill_approx(vs, q, pc->dim, m, -1.0f, s), 0);
		const rec_ref_t *got = rec_set_at(s);
		size_t cnt = rec_set_count(s);
		for (size_t i = 0; i < k; i++) {
			int found = 0;
			for (size_t j = 0; j < cnt; j++)
				if (got[j] == bf[i].ref)
					found = 1;
			ASSERT(found, "brute-force top-k ⊆ fill set (recall@k=1)");
			if (!found)
				break;
		}
		ASSERT(rec_set_approx(s) == REC_SET_APPROX, "fill is approximate");
		rec_set_free(s);
	}

	sepal_close(vs);
	free(base);
	free(q);
	free(v);
}

static void
test_recall_prop(void)
{
	const uint64_t seeds[] = { 1, 42, 1337 };
	const prop_case_t cases[] = {
		{ .n_store = 64,   .dim = 64,   .n_query = 250 },
		{ .n_store = 256,  .dim = 256,  .n_query = 200 },
		{ .n_store = 400,  .dim = 384,  .n_query = 150 },
		{ .n_store = 1000, .dim = 768,  .n_query = 100 },
		{ .n_store = 3000, .dim = 768,  .n_query = 40  },
	};

	printf("=== recall@k ≡ 1.0 (seeds × dims × sizes) ===\n");
	for (size_t si = 0; si < sizeof(seeds) / sizeof(seeds[0]); si++) {
		for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
			printf("  seed %llu, n=%zu dim=%zu ...\n",
			       (unsigned long long)seeds[si],
			       cases[ci].n_store, cases[ci].dim);
			run_case(&cases[ci], seeds[si]);
		}
	}
}

int
main(void)
{
	test_recall_prop();
	return test_summary();
}