/*
 * test_axis_store.c — 2A-1: rec_axis_store / rec_axis_unstore /
 * rec_axis_readback adapter contract for libsepal.
 *
 * Dual grammar (PHASE-2-CLI.md §2A, per-axis role "libsepal"):
 *   "f1,f2,…"  comma floats (dim = token count, 1..SEPAL_VEC_MAX) → direct
 *   else       embedded via the configured embedder, or EINVAL.
 *
 * The curl call in sepal_embed_fetch is replaced here with a canned-vector
 * stub (strong symbol overriding the library's weak default) so the suite
 * stays offline-green. A live curl test lives separately in
 * test_axis_store_live.c (env-gated, skipped by default).
 */

#include "../test_common.h"
#include <ttypt/rec.h>

#include <errno.h>

/* ── embed stub: executable-global strong symbol overrides libsepal's weak
 * default, so every rec_axis_store embed path in this binary returns the
 * canned 3-float vector. Honors a NULL out (vector must still be counted). */

static float stub_vec[] = { 0.5f, -0.25f, 0.75f };

int
sepal_embed_fetch(const char *text, float **vec_out, size_t *n_out)
{
	float *v;

	(void)text;
	*vec_out = NULL;
	*n_out = 0;
	if (!vec_out || !n_out)
		return -1;
	v = malloc(sizeof(stub_vec));
	if (!v)
		return -1;
	memcpy(v, stub_vec, sizeof(stub_vec));
	*vec_out = v;
	*n_out = 3;
	return 0;
}

static void
test_store_floats_roundtrip(void)
{
	printf("=== store: floats → sepal_get cross-check ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	ASSERT_EQ(rec_axis_store(vs, NULL, 1, "1.0,2.0,3.0"), 0);

	float out[3] = { 0.0f, 0.0f, 0.0f };
	ASSERT_EQ(sepal_get(vs, 1, out, 3), 3);
	ASSERT_NEAR(out[0], 1.0f, 1e-6f);
	ASSERT_NEAR(out[1], 2.0f, 1e-6f);
	ASSERT_NEAR(out[2], 3.0f, 1e-6f);
	ASSERT_EQ(sepal_dim(vs, 1), 3);
	sepal_close(vs);
}

static void
test_readback_matches_store(void)
{
	printf("=== readback: malloc'd comma-floats of the stored vector ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(rec_axis_store(vs, NULL, 7, "0.5,-1.25,4.0"), 0);

	char *blob = NULL;
	size_t n = 0;
	ASSERT_EQ(rec_axis_readback(vs, 7, &blob, &n), 0);
	ASSERT_NOT_NULL(blob);
	ASSERT(n == strlen(blob), "n_out counts the display chars");

	/* re-parse the readback string, must recover the same floats */
	{
		char *copy = strdup(blob);
		ASSERT_NOT_NULL(copy);
		char *save = NULL;
		float expect[] = { 0.5f, -1.25f, 4.0f };
		int i = 0;
		for (char *tok = strtok_r(copy, ",", &save); tok;
		     tok = strtok_r(NULL, ",", &save)) {
			ASSERT(i < 3, "three tokens from readback");
			ASSERT_NEAR((float)strtod(tok, NULL), expect[i], 1e-6f);
			i++;
		}
		ASSERT_EQ(i, 3);
		free(copy);
	}
	free(blob);

	/* absent ref reads back NULL/0, still returns 0 */
	blob = (char *)0x1;
	n = 9;
	ASSERT_EQ(rec_axis_readback(vs, 99, &blob, &n), 0);
	ASSERT_NULL(blob);
	ASSERT_EQ(n, (size_t)0);
	sepal_close(vs);
}

static void
test_unstore_idempotent(void)
{
	printf("=== unstore: entries gone; absent is idempotent 0 ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(rec_axis_store(vs, NULL, 3, "9.0,8.0"), 0);

	ASSERT_EQ(rec_axis_unstore(vs, 3), 0);
	ASSERT_EQ(sepal_dim(vs, 3), 0);
	{
		char *blob = (char *)0x1;
		size_t n = 1;
		ASSERT_EQ(rec_axis_readback(vs, 3, &blob, &n), 0);
		ASSERT_NULL(blob);
		ASSERT_EQ(n, (size_t)0);
	}

	ASSERT_EQ(rec_axis_unstore(vs, 3), 0);
	ASSERT_EQ(rec_axis_unstore(vs, 31337), 0);
	sepal_close(vs);
}

static void
test_replace_in_place(void)
{
	printf("=== store: same ref replaces in place ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(rec_axis_store(vs, NULL, 1, "1.0,2.0"), 0);
	ASSERT_EQ(sepal_n(vs), 1);
	ASSERT_EQ(rec_axis_store(vs, NULL, 1, "3.0,4.0,5.0"), 0);
	ASSERT_EQ(sepal_n(vs), 1);

	float out[3] = { 0.0f, 0.0f, 0.0f };
	ASSERT_EQ(sepal_get(vs, 1, out, 3), 3);
	ASSERT_NEAR(out[0], 3.0f, 1e-6f);
	ASSERT_NEAR(out[1], 4.0f, 1e-6f);
	ASSERT_NEAR(out[2], 5.0f, 1e-6f);
	sepal_close(vs);
}

static void
test_bad_grammar(void)
{
	printf("=== store: empty / non-float without embedder → EINVAL ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);

	errno = 0;
	ASSERT_EQ(rec_axis_store(vs, NULL, 1, ""), -1);
	ASSERT_EQ(errno, EINVAL);

	errno = 0;
	ASSERT_EQ(rec_axis_store(vs, NULL, 1, "not-a-float-list"), -1);
	ASSERT_EQ(errno, EINVAL);

	/* a partially-float string also falls through and is rejected */
	errno = 0;
	ASSERT_EQ(rec_axis_store(vs, NULL, 1, "1.0,abc"), -1);
	ASSERT_EQ(errno, EINVAL);
	sepal_close(vs);
}

static void
test_over_cap(void)
{
	printf("=== store: float list over SEPAL_VEC_MAX → ERANGE ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);

	/* build a string with SEPAL_VEC_MAX+3 tokens */
	size_t want = SEPAL_VEC_MAX + 3, cap = want * 4 + 1, used = 0;
	char *s = malloc(cap);
	ASSERT_NOT_NULL(s);
	s[0] = '\0';
	for (size_t i = 0; i < want; i++) {
		used += (size_t)snprintf(s + used, cap - used, i ? ",1" : "1");
	}
	errno = 0;
	ASSERT_EQ(rec_axis_store(vs, NULL, 1, s), -1);
	ASSERT_EQ(errno, ERANGE);
	free(s);
	sepal_close(vs);
}

static void
test_embed_path_via_stub(void)
{
	printf("=== store: string + configured embedder → stub vector ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(sepal_configure_embeddings("http://localhost:9/none",
	                                     "test-model", NULL), 0);

	ASSERT_EQ(rec_axis_store(vs, NULL, 11, "hello world"), 0);

	float out[3] = { 0.0f, 0.0f, 0.0f };
	ASSERT_EQ(sepal_get(vs, 11, out, 3), 3);
	ASSERT_NEAR(out[0], 0.5f, 1e-6f);
	ASSERT_NEAR(out[1], -0.25f, 1e-6f);
	ASSERT_NEAR(out[2], 0.75f, 1e-6f);

	/* readback of an embedded vector renders as comma floats too */
	{
		char *blob = NULL;
		size_t n = 0;
		ASSERT_EQ(rec_axis_readback(vs, 11, &blob, &n), 0);
		ASSERT_NOT_NULL(blob);
		ASSERT(strchr(blob, ',') != NULL, "comma-joined readback");
		free(blob);
	}

	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);
	sepal_close(vs);
}

static void
test_null_ctx(void)
{
	printf("=== store/unstore/readback: NULL ctx → -1 ===\n");
	char *blob = (char *)0x1;
	size_t n = 1;
	ASSERT_EQ(rec_axis_store(NULL, NULL, 1, "1.0"), -1);
	ASSERT_EQ(rec_axis_unstore(NULL, 1), -1);
	ASSERT_EQ(rec_axis_readback(NULL, 1, &blob, &n), -1);
	ASSERT_NULL(blob);
	ASSERT_EQ(n, (size_t)0);
}

int
main(void)
{
	test_store_floats_roundtrip();
	test_readback_matches_store();
	test_unstore_idempotent();
	test_replace_in_place();
	test_bad_grammar();
	test_over_cap();
	test_embed_path_via_stub();
	test_null_ctx();
	return test_summary();
}
