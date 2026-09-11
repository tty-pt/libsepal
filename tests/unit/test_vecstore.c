/* test_vecstore.c — vecstore lifecycle: put/get/del/dim/n, opaque refs. */

#include "../test_common.h"

#define CHK_CAP 4096

static void
test_lifecycle(void)
{
	printf("=== vecstore: put/get/del/n lifecycle ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(sepal_n(vs), 0);

	float v[64], w[64];
	sepal_rng_t r = { 21 };
	rng_unit_vector(&r, v, 64);
	rng_unit_vector(&r, w, 64);

	ASSERT_EQ(sepal_put(vs, 11, v, 64), 0);
	ASSERT_EQ(sepal_put(vs, 22, w, 64), 0);
	ASSERT_EQ(sepal_n(vs), 2);

	float out[CHK_CAP];
	ASSERT_EQ(sepal_get(vs, 11, out, CHK_CAP), 64);
	ASSERT(memcmp(out, v, 64 * sizeof(float)) == 0, "v round-trip");
	ASSERT_EQ(sepal_get(vs, 22, out, CHK_CAP), 64);
	ASSERT(memcmp(out, w, 64 * sizeof(float)) == 0, "w round-trip");

	ASSERT_EQ(sepal_dim(vs, 11), 64);
	ASSERT_EQ(sepal_full_dim(vs, 11), 64);

	ASSERT_EQ(sepal_del(vs, 11), 0);
	ASSERT_EQ(sepal_n(vs), 1);
	ASSERT_EQ(sepal_get(vs, 11, out, CHK_CAP), 0);      /* gone */
	ASSERT_EQ(sepal_dim(vs, 11), 0);
	ASSERT_EQ(sepal_del(vs, 11), -1);                   /* absent */
	ASSERT_EQ(sepal_del(vs, 99), -1);
	sepal_close(vs);
}

static void
test_reput_replaces(void)
{
	printf("=== vecstore: re-put same ref replaces ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float a[16], b[16];
	sepal_rng_t r = { 22 };
	rng_unit_vector(&r, a, 16);
	rng_unit_vector(&r, b, 16);

	ASSERT_EQ(sepal_put(vs, 5, a, 16), 0);
	ASSERT_EQ(sepal_n(vs), 1);
	ASSERT_EQ(sepal_put(vs, 5, b, 16), 0);              /* replaces, not adds */
	ASSERT_EQ(sepal_n(vs), 1);

	float out[CHK_CAP];
	ASSERT_EQ(sepal_get(vs, 5, out, CHK_CAP), 16);
	ASSERT(memcmp(out, b, 16 * sizeof(float)) == 0, "replaced blob");
	sepal_close(vs);
}

static void
test_opaque_refs(void)
{
	printf("=== vecstore: refs are opaque u64 ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float v[32];
	sepal_rng_t r = { 23 };
	rng_unit_vector(&r, v, 32);

	rec_ref_t refs[] = { 0, 1, 0xFFFFULL, 1ULL << 63, (rec_ref_t)-1 };
	for (size_t i = 0; i < sizeof(refs) / sizeof(refs[0]); i++)
		ASSERT_EQ(sepal_put(vs, refs[i], v, 32), 0);
	ASSERT_EQ(sepal_n(vs), 5);

	float out[CHK_CAP];
	ASSERT_EQ(sepal_get(vs, 1ULL << 63, out, CHK_CAP), 32);
	ASSERT_EQ(sepal_get(vs, (rec_ref_t)-1, out, CHK_CAP), 32);
	ASSERT_EQ(sepal_get(vs, 0, out, CHK_CAP), 32);
	sepal_close(vs);
}

static void
test_null_guards(void)
{
	printf("=== vecstore: NULL guards ===\n");
	float v[8] = { 0 };
	ASSERT_EQ(sepal_put(NULL, 1, v, 8), -1);
	ASSERT_EQ(sepal_del(NULL, 1), -1);
	ASSERT_EQ(sepal_get(NULL, 1, v, 8), 0);
	ASSERT_EQ(sepal_dim(NULL, 1), 0);
	ASSERT_EQ(sepal_full_dim(NULL, 1), 0);
	ASSERT_EQ(sepal_n(NULL), 0);
}

int
main(void)
{
	test_lifecycle();
	test_reput_replaces();
	test_opaque_refs();
	test_null_guards();
	return test_summary();
}