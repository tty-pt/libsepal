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
	test_rank_missing_below_threshold();
	return test_summary();
}