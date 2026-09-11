/* test_matryoshka.c — matryoshka truncation + dim invariants (D9). */

#include "../test_common.h"

#define CHK_CAP 4096

static void
test_truncation_768(void)
{
	printf("=== matryoshka: 768 → exact 256 prefix, bit-exact ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float v[768];
	sepal_rng_t r = { 11 };
	rng_unit_vector(&r, v, 768);

	ASSERT_EQ(sepal_put(vs, 7, v, 768), 0);
	ASSERT_EQ(sepal_dim(vs, 7), 256);
	ASSERT_EQ(sepal_full_dim(vs, 7), 768);

	float out[CHK_CAP];
	ASSERT_EQ(sepal_get(vs, 7, out, CHK_CAP), 256);
	ASSERT(memcmp(out, v, 256 * sizeof(float)) == 0, "prefix floats bit-exact");

	ASSERT_EQ(sepal_get(vs, 7, out, 256), 256);      /* exactly dim ok */
	sepal_close(vs);
}

static void
test_no_padding_small(void)
{
	printf("=== matryoshka: vectors ≤ 256 stored untruncated ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float v[128];
	sepal_rng_t r = { 12 };
	rng_unit_vector(&r, v, 128);

	ASSERT_EQ(sepal_put(vs, 1, v, 128), 0);
	ASSERT_EQ(sepal_dim(vs, 1), 128);
	ASSERT_EQ(sepal_full_dim(vs, 1), 128);

	float out[CHK_CAP];
	ASSERT_EQ(sepal_get(vs, 1, out, CHK_CAP), 128);
	ASSERT(memcmp(out, v, 128 * sizeof(float)) == 0, "128-dim bit-exact");

	/* dim 256 fresh store too */
	float v2[256];
	rng_unit_vector(&r, v2, 256);
	ASSERT_EQ(sepal_put(vs, 2, v2, 256), 0);
	ASSERT_EQ(sepal_dim(vs, 2), 256);
	ASSERT_EQ(sepal_full_dim(vs, 2), 256);
	sepal_close(vs);
}

static void
test_dim_validation(void)
{
	printf("=== matryoshka: put dimension validation ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float v[8];
	sepal_rng_t r = { 13 };
	rng_unit_vector(&r, v, 8);

	ASSERT_EQ(sepal_put(vs, 1, v, 0), -1);              /* full_dim 0 */
	ASSERT_EQ(sepal_put(vs, 1, v, SEPAL_VEC_MAX + 1), -1); /* over max */
	ASSERT_EQ(sepal_put(vs, 1, NULL, 8), -1);
	ASSERT_EQ(sepal_put(NULL, 1, v, 8), -1);
	ASSERT_EQ(sepal_put(vs, 1, v, 8), 0);               /* still usable */
	sepal_close(vs);
}

static void
test_get_never_truncates(void)
{
	printf("=== matryoshka: get over-max → 0 (never truncate) ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float v[768];
	sepal_rng_t r = { 14 };
	rng_unit_vector(&r, v, 768);
	ASSERT_EQ(sepal_put(vs, 9, v, 768), 0);

	float out[CHK_CAP];
	ASSERT_EQ(sepal_get(vs, 9, out, 100), 0);           /* mismatch → 0 */
	ASSERT_EQ(sepal_get(vs, 9, out, 256), 256);         /* exact ok */
	ASSERT_EQ(sepal_get(vs, 9, out, 0), 0);             /* max 0 */
	sepal_close(vs);
}

int
main(void)
{
	test_truncation_768();
	test_no_padding_small();
	test_dim_validation();
	test_get_never_truncates();
	return test_summary();
}