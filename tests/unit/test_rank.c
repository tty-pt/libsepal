/* test_rank.c — sepal_rank: exact cosine rerank callback (rec_score_fn). */

#include "../test_common.h"

static void
test_rank_matches_cosine(void)
{
	printf("=== rank: == exact cosine for stored refs ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float v[128], q[128];
	sepal_rng_t r = { 51 };
	rng_unit_vector(&r, v, 128);
	rng_unit_vector(&r, q, 128);
	ASSERT_EQ(sepal_put(vs, 42, v, 128), 0);

	struct sepal_rank_ctx ctx = { vs, q, 128, -1.0f };
	float score = -1.0f;
	ASSERT_EQ(sepal_rank(&ctx, 42, &score), 0);
	ASSERT_NEAR(score, sepal_cosine(v, q, 128), 1e-6f);

	/* full_dim 256 store, q longer than exact dim → still ranked on the
	 * truncated prefix */
	float v2[256], q2[768];
	rng_unit_vector(&r, v2, 256);
	rng_unit_vector(&r, q2, 768);
	ASSERT_EQ(sepal_put(vs, 7, v2, 256), 0);
	struct sepal_rank_ctx ctx2 = { vs, q2, 768, -1.0f };
	ASSERT_EQ(sepal_rank(&ctx2, 7, &score), 0);
	ASSERT_NEAR(score, sepal_cosine(v2, q2, 256), 1e-4f);
	sepal_close(vs);
}

static void
test_rank_score_equiv_dimensions(void)
{
	printf("=== rank: stage-2 score == sepal_cosine across dims/norms ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	static const size_t dims[] = { 1, 2, 7, 63, 64, 65, 128, 256 };
	sepal_rng_t r = { 101 };
	for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++) {
		size_t d = dims[i];
		float *vu = malloc(d * sizeof(float));
		float *v = malloc(d * sizeof(float));
		float *q = malloc(d * sizeof(float));
		ASSERT_NOT_NULL(vu); ASSERT_NOT_NULL(v); ASSERT_NOT_NULL(q);
		/* unit vector (stored norm == 1) */
		rng_unit_vector(&r, vu, d);
		rng_unit_vector(&r, q, d);
		ASSERT_EQ(sepal_put(vs, (rec_ref_t)(d * 2 + 1), vu, d), 0);
		struct sepal_rank_ctx c = { vs, q, d, -1.0f };
		float sc = -1.0f;
		ASSERT_EQ(sepal_rank(&c, (rec_ref_t)(d * 2 + 1), &sc), 0);
		ASSERT_NEAR(sc, sepal_cosine(vu, q, d), 1e-6f);

		/* scaled vector (stored norm != 1) exercises the frozen norm */
		for (size_t j = 0; j < d; j++)
			v[j] = (float)rng_unit(&r) * 3.0f;
		ASSERT_EQ(sepal_put(vs, (rec_ref_t)(d * 2 + 2), v, d), 0);
		ASSERT_EQ(sepal_rank(&c, (rec_ref_t)(d * 2 + 2), &sc), 0);
		ASSERT_NEAR(sc, sepal_cosine(v, q, d), 1e-6f);

		/* q longer than stored dim: score over the stored prefix */
		float *qlong = malloc((d + 32) * sizeof(float));
		rng_unit_vector(&r, qlong, d + 32);
		struct sepal_rank_ctx c2 = { vs, qlong, d + 32, -1.0f };
		float sl = -1.0f;
		ASSERT_EQ(sepal_rank(&c2, (rec_ref_t)(d * 2 + 1), &sl), 0);
		ASSERT_NEAR(sl, sepal_cosine(vu, qlong, d), 1e-6f);
		free(qlong);
		free(q);
		free(v);
		free(vu);
	}
	sepal_close(vs);
}

static void
test_rank_missing_below_threshold(void)
{
	printf("=== rank: missing ref / min_sim / dims mismatch → nonzero ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float v[64], q[64];
	sepal_rng_t r = { 52 };
	rng_unit_vector(&r, v, 64);
	rng_unit_vector(&r, q, 64);
	ASSERT_EQ(sepal_put(vs, 1, v, 64), 0);

	/* missing ref: nonzero, score untouched */
	struct sepal_rank_ctx ctx = { vs, q, 64, 0.0f };
	float score = 1.234f;
	ASSERT_NE(sepal_rank(&ctx, 999, &score), 0);
	ASSERT_NEAR(score, 1.234f, 0.0f);

	/* below min_sim: q = far-away vector */
	float far[64];
	memset(far, 0, sizeof(far));
	far[0] = 1.0f;                                  /* orthogonal to v[0..n) */
	struct sepal_rank_ctx ctx2 = { vs, far, 64, 0.5f };
	ASSERT_NE(sepal_rank(&ctx2, 1, &score), 0);

	/* qdim too small vs stored dim */
	struct sepal_rank_ctx ctx3 = { vs, q, 32, 0.0f };
	ASSERT_NE(sepal_rank(&ctx3, 1, &score), 0);

	/* NULL guards */
	ASSERT_NE(sepal_rank(NULL, 1, &score), 0);
	struct sepal_rank_ctx ctx4 = { vs, NULL, 64, 0.0f };
	ASSERT_NE(sepal_rank(&ctx4, 1, &score), 0);
	struct sepal_rank_ctx ctx5 = { vs, q, 0, 0.0f };
	ASSERT_NE(sepal_rank(&ctx5, 1, &score), 0);
	ASSERT_NE(sepal_rank(&ctx, 1, NULL), 0);
	sepal_close(vs);
}

int
main(void)
{
	test_rank_matches_cosine();
	test_rank_score_equiv_dimensions();
	test_rank_missing_below_threshold();
	return test_summary();
}