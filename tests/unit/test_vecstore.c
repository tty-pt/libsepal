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
	printf("=== vecstore: refs are opaque u32 ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float v[32];
	sepal_rng_t r = { 23 };
	rng_unit_vector(&r, v, 32);

	rec_ref_t refs[] = { 0, 1, 0xFFFF, 0x7FFFFFFF, UINT32_MAX };
	for (size_t i = 0; i < sizeof(refs) / sizeof(refs[0]); i++)
		ASSERT_EQ(sepal_put(vs, refs[i], v, 32), 0);
	ASSERT_EQ(sepal_n(vs), 5);

	float out[CHK_CAP];
	ASSERT_EQ(sepal_get(vs, 0x7FFFFFFF, out, CHK_CAP), 32);
	ASSERT_EQ(sepal_get(vs, UINT32_MAX, out, CHK_CAP), 32);
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

static void
test_index_valid_after_mutation(void)
{
	printf("=== vecstore: index validate 0 across scripted+random mutation ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(sepal_index_validate(vs), 0);

	float v[300];
	sepal_rng_t r = { 77 };
	for (size_t it = 0; it < 3; it++) {          /* 3 mutation passes */
		size_t base = it * 3000;
		for (size_t i = 0; i < 300; i++) {
			rng_unit_vector(&r, v, 128);
			ASSERT_EQ(sepal_put(vs, (rec_ref_t)(base + i), v, 128), 0);
		}
		for (size_t i = 0; i < 100; i++) {       /* rebuild larger dims */
			rng_unit_vector(&r, v, 300);
			ASSERT_EQ(sepal_put(vs, (rec_ref_t)(base + 20 + i * 2), v, 300), 0);
		}
		for (size_t i = 0; i < 60; i++)         /* deletes */
			ASSERT_EQ(sepal_del(vs, (rec_ref_t)(base + i * 3)), 0);
		ASSERT_EQ(sepal_index_validate(vs), 0);
	}
	ASSERT_EQ(sepal_n(vs), 720);        /* 3 passes × (300 puts − 60 dels) */
	ASSERT_EQ(sepal_index_validate(vs), 0);
	sepal_close(vs);
}

static void
test_index_search_parity_after_mutation(void)
{
	printf("=== vecstore: search parity (vs brute) after mutation ===\n");
	const size_t DIM = 192;
	const rec_ref_t MAXR = 108;
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float v[DIM];
	sepal_rng_t r = { 78 };
	for (size_t i = 0; i < 60; i++) {
		rng_unit_vector(&r, v, DIM);
		ASSERT_EQ(sepal_put(vs, (rec_ref_t)i, v, DIM), 0);
	}
	for (size_t i = 0; i < 15; i++) {          /* some replaces */
		rng_unit_vector(&r, v, DIM);
		ASSERT_EQ(sepal_put(vs, (rec_ref_t)(i * 2), v, DIM), 0);
	}
	for (size_t i = 0; i < 20; i++)            /* some deletes */
		ASSERT_EQ(sepal_del(vs, (rec_ref_t)(i * 3 + 1)), 0);
	for (size_t i = 0; i < 8; i++) {           /* fresh refs beyond 60 */
		rng_unit_vector(&r, v, DIM);
		ASSERT_EQ(sepal_put(vs, (rec_ref_t)(100 + i), v, DIM), 0);
	}
	ASSERT_EQ(sepal_index_validate(vs), 0);

	/* dump the surviving store via sepal_get into a brute-force base */
	float *base = malloc(MAXR * DIM * sizeof(float));
	rec_ref_t survive[MAXR];
	size_t nsur = 0;
	float out[DIM];
	for (rec_ref_t rf = 0; rf < MAXR; rf++) {
		if (sepal_get(vs, rf, out, DIM) == DIM) {
			memcpy(base + nsur * DIM, out, DIM * sizeof(float));
			survive[nsur++] = rf;
		}
	}
	ASSERT_EQ(nsur, sepal_n(vs));

	sepal_bf_hit_t bf[16];
	sepal_hit_t sh[16];
	for (size_t qi = 0; qi < 30; qi++) {
		rng_unit_vector(&r, v, DIM);
		size_t bfn = brute_force_search(base, nsur, DIM, 256, v, DIM,
		                                8, 0.0f, bf);
		for (size_t j = 0; j < bfn; j++)  /* brute ref = index into survive[] */
			bf[j].ref = survive[bf[j].ref];
		size_t shn = sepal_search(vs, v, DIM, 8, 0.0f, nsur, sh);
		ASSERT_EQ(shn, bfn);
		int inse[16] = { 0 };
		for (size_t i = 0; i < shn; i++) {
			for (size_t j = 0; j < bfn; j++)
				if (sh[i].ref == bf[j].ref) {
					inse[i] = 1;
					ASSERT_NEAR(sh[i].score, bf[j].score, 1e-6f);
				}
			ASSERT(inse[i], "hit in brute set");
		}
		for (size_t i = 1; i < shn; i++)
			ASSERT(sh[i - 1].score >= sh[i].score, "non-increasing");
	}
	free(base);
	sepal_close(vs);
}

int
main(void)
{
	test_lifecycle();
	test_reput_replaces();
	test_opaque_refs();
	test_null_guards();
	test_index_valid_after_mutation();
	test_index_search_parity_after_mutation();
	return test_summary();
}