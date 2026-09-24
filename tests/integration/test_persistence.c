/* test_persistence.c — file-backed store: close → reopen → identical. */

#include "../test_common.h"

#include <unistd.h>

#define N_VEC    200
#define DIM      768
#define CHK_CAP  4096

static const char *
tmp_path(char *buf, size_t n)
{
	snprintf(buf, n, "/tmp/test_sepal_persist_%ld.db", (long)getpid());
	return buf;
}

static void
test_persistence_roundtrip(void)
{
	printf("=== persistence: put N → close → reopen → identical ===\n");

	char path[256];
	tmp_path(path, sizeof(path));

	sepal_vecstore_t *vs = sepal_open(path, NULL);
	ASSERT_NOT_NULL(vs);

	/* fixed seed so the reopen comparison is byte-for-byte meaningful */
	sepal_rng_t r = { 61 };
	float *v = malloc(DIM * sizeof(float));
	ASSERT_NOT_NULL(v);
	for (size_t i = 0; i < N_VEC; i++) {
		rng_unit_vector(&r, v, DIM);
		ASSERT_EQ(sepal_put(vs, (rec_ref_t)i, v, DIM), 0);
	}
	ASSERT_EQ(sepal_n(vs), N_VEC);
	sepal_close(vs);

	vs = sepal_open(path, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(sepal_n(vs), N_VEC);

	float *got = malloc(DIM * sizeof(float));
	ASSERT_NOT_NULL(got);
	r = (sepal_rng_t){ 61 };
	for (size_t i = 0; i < N_VEC; i++) {
		rng_unit_vector(&r, v, DIM);          /* regenerated reference */
		ASSERT_EQ(sepal_dim(vs, (rec_ref_t)i), 256);
		ASSERT_EQ(sepal_full_dim(vs, (rec_ref_t)i), DIM);
		ASSERT_EQ(sepal_get(vs, (rec_ref_t)i, got, CHK_CAP), 256);
		ASSERT(memcmp(got, v, 256 * sizeof(float)) == 0, "blob floats identical");
	}

	/* search parity: same top-k set after reopen as before (brute-ref) */
	r = (sepal_rng_t){ 62 };
	float q[DIM];
	float *base = malloc(N_VEC * DIM * sizeof(float));
	ASSERT_NOT_NULL(base);
	for (size_t i = 0; i < N_VEC; i++)
		sepal_get(vs, (rec_ref_t)i, base + i * DIM, CHK_CAP);

	for (size_t qi = 0; qi < 50; qi++) {
		rng_unit_vector(&r, q, DIM);
		sepal_hit_t sh[8];
		size_t shn = sepal_search(vs, q, DIM, 8, 0.05f, N_VEC, sh);
		sepal_bf_hit_t bf[8];
		size_t bfn = brute_force_search(base, N_VEC, DIM, 256, q, DIM, 8, 0.05f, bf);
		ASSERT_EQ(shn, bfn);
		for (size_t i = 0; i < shn && i < 8; i++) {
			ASSERT_EQ(sh[i].ref, bf[i].ref);
			ASSERT_NEAR(sh[i].score, bf[i].score, 1e-4f);
		}
	}

	sepal_close(vs);
	unlink(path);
	free(base);
	free(got);
	free(v);
}

int
main(void)
{
	test_persistence_roundtrip();
	return test_summary();
}