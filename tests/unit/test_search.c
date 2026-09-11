/* test_search.c — two-stage ANN search vs exact brute-force reference. */

#include "../test_common.h"

#define CORPUS_N   200
#define QUERY_N    50
#define CORPUS_DIM 768
#define CHK_CAP    4096

static sepal_vecstore_t *
build_corpus(size_t n, uint64_t seed)
{
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	if (!vs)
		return NULL;
	float *v = malloc(CORPUS_DIM * sizeof(float));
	if (!v) {
		sepal_close(vs);
		return NULL;
	}
	sepal_rng_t r = { seed };
	for (size_t i = 0; i < n; i++) {
		rng_unit_vector(&r, v, CORPUS_DIM);
		sepal_put(vs, (rec_ref_t)i, v, CORPUS_DIM);
	}
	free(v);
	return vs;
}

static void
test_search_parity(void)
{
	printf("=== search: == brute force with full pool (m=n) ===\n");
	sepal_vecstore_t *vs = build_corpus(CORPUS_N, 31);
	ASSERT_NOT_NULL(vs);

	size_t k_list[] = { 1, 5, 10 };
	float min_list[] = { 0.0f, 0.2f };

	sepal_rng_t r = { 32 };
	float q[CORPUS_DIM];
	float *base = malloc(CORPUS_N * CORPUS_DIM * sizeof(float));
	ASSERT_NOT_NULL(base);
	for (size_t i = 0; i < CORPUS_N; i++)
		sepal_get(vs, (rec_ref_t)i, base + i * CORPUS_DIM, CHK_CAP);

	for (size_t qi = 0; qi < QUERY_N; qi++) {
		rng_unit_vector(&r, q, CORPUS_DIM);
		for (size_t ki = 0; ki < sizeof(k_list) / sizeof(k_list[0]); ki++) {
			for (size_t mi = 0; mi < sizeof(min_list) / sizeof(min_list[0]); mi++) {
				size_t k = k_list[ki];
				float mins = min_list[mi];

				sepal_hit_t sh[16];
				size_t shn = sepal_search(vs, q, CORPUS_DIM, k, mins,
				                          CORPUS_N, sh);

				sepal_bf_hit_t bf[16];
size_t bfn = brute_force_search(base, CORPUS_N,
				                                CORPUS_DIM, 256,
				                                q, CORPUS_DIM, k, mins, bf);

				ASSERT_EQ(shn, bfn);
				size_t cmp = shn < k ? shn : k;
				for (size_t i = 0; i < cmp; i++) {
					ASSERT_EQ(sh[i].ref, bf[i].ref);
					ASSERT_NEAR(sh[i].score, bf[i].score, 1e-4);
				}
				/* non-increasing scores, everything above min_sim */
				for (size_t i = 1; i < shn; i++)
					ASSERT(sh[i - 1].score >= sh[i].score, "scores non-increasing");
				for (size_t i = 0; i < shn; i++)
					ASSERT(sh[i].score + 1e-6f >= mins, "all above min_sim");
			}
		}
	}
	free(base);
	sepal_close(vs);
}

static void
test_search_ordering_crafted(void)
{
	printf("=== search: order = cosine desc (ties none, refs asc) ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float e[768];
	memset(e, 0, sizeof(e));
	for (size_t i = 0; i < 4; i++) {
		e[i] = 1.0f;
		ASSERT_EQ(sepal_put(vs, (rec_ref_t)(i + 1), e, 768), 0);
		memset(e, 0, sizeof(e));
	}

	float q[768];
	memset(q, 0, sizeof(q));
	q[0] = 5.0f; q[1] = 3.0f; q[2] = 1.0f; q[3] = 0.5f;
	/* normalize to unit so cos(q, e_i) = q_i / |q| */
	double nrm = 0;
	for (size_t i = 0; i < 4; i++)
		nrm += (double)q[i] * (double)q[i];
	nrm = sqrt(nrm);
	for (size_t i = 0; i < 4; i++)
		q[i] = (float)((double)q[i] / nrm);

	sepal_hit_t hits[8];
	size_t h = sepal_search(vs, q, 768, 4, 0.05f, 4, hits);
	ASSERT_EQ(h, 4);
	ASSERT_EQ(hits[0].ref, 1);
	ASSERT_EQ(hits[1].ref, 2);
	ASSERT_EQ(hits[2].ref, 3);
	ASSERT_EQ(hits[3].ref, 4);
	ASSERT(hits[0].score > hits[1].score, "e1 best");
	sepal_close(vs);
}

static void
test_search_knobs(void)
{
	printf("=== search: k/m/min_sim knobs + guards ===\n");
	sepal_vecstore_t *vs = build_corpus(50, 33);
	ASSERT_NOT_NULL(vs);

	float q[768];
	sepal_rng_t r = { 34 };
	rng_unit_vector(&r, q, 768);
	sepal_hit_t hits[16];

	/* m < k caps candidates */
	size_t h = sepal_search(vs, q, 768, 10, 0.0f, 2, hits);
	ASSERT(h <= 2, "m<k → at most m hits");

	/* m == 0 → default 10*k capped at n (n=50) */
	h = sepal_search(vs, q, 768, 3, 0.0f, 0, hits);
	ASSERT(h <= 3, "default m still bounded by k");
	for (size_t i = 1; i < h; i++)
		ASSERT(hits[i - 1].score >= hits[i].score, "non-increasing");

	/* guards */
	ASSERT_EQ(sepal_search(NULL, q, 768, 5, 0.0f, 0, hits), 0);
	ASSERT_EQ(sepal_search(vs, NULL, 768, 5, 0.0f, 0, hits), 0);
	ASSERT_EQ(sepal_search(vs, q, 0, 5, 0.0f, 0, hits), 0);
	ASSERT_EQ(sepal_search(vs, q, 768, 0, 0.0f, 0, hits), 0);
	ASSERT_EQ(sepal_search(vs, q, 768, 5, 0.0f, 0, NULL), 0);
	ASSERT_EQ(sepal_search(vs, q, SEPAL_VEC_MAX + 1, 5, 0.0f, 0, hits), 0);
	sepal_close(vs);
}

static void
test_search_empty_store(void)
{
	printf("=== search: empty / tiny stores ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float q[64];
	sepal_rng_t r = { 35 };
	rng_unit_vector(&r, q, 64);
	sepal_hit_t hits[8];
	ASSERT_EQ(sepal_search(vs, q, 64, 5, 0.0f, 0, hits), 0);
	sepal_close(vs);
}

int
main(void)
{
	test_search_parity();
	test_search_ordering_crafted();
	test_search_knobs();
	test_search_empty_store();
	return test_summary();
}