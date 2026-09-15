/*
 * test_axis_decode_query.c — Phase 6: sepal query-time `query=` leaf.
 *
 * `sepal="query='hello world' m=10 min_sim=0.2"` embeds the string
 * server-side in sepal_axis_decode (stoma single-quote parity) instead of
 * the pi-mm curl→tempfile→`file=` bridge. Precedence: non-empty `query`
 * wins over `file=`; empty query + no file → NULL; unconfigured embedder
 * → NULL without calling fetch.
 *
 * The curl call in sepal_embed_fetch is replaced here with a canned-vector
 * stub (strong symbol overriding the library's weak default, same pattern
 * as test_axis_store.c) so the suite stays offline-green.
 */

#include "../test_common.h"
#include <ttypt/rec.h>
#include <unistd.h>

/* ── embed stub: canned 3-float vector, records received text + call count */

static float stub_vec[] = { 0.5f, -0.25f, 0.75f };
static char stub_text[256];
static int stub_calls;

int
sepal_embed_fetch(const char *text, float **vec_out, size_t *n_out)
{
	float *v;

	stub_calls++;
	snprintf(stub_text, sizeof(stub_text), "%s", text ? text : "");
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

static int
find_sepal_slot(void)
{
	int i;
	const rec_axis_t *axis;

	for (i = 0; i < rec_axis_count(); i++) {
		axis = rec_axis_get(i);
		if (axis && !strcmp(axis->name, "sepal"))
			return i;
	}
	return -1;
}

static void
stub_reset(void)
{
	stub_calls = 0;
	stub_text[0] = '\0';
}

static void
test_decode_query_quoted(void)
{
	printf("=== decode: query='hello world' embeds verbatim ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	ASSERT_EQ(sepal_configure_embeddings("http://localhost:9/none",
	                                     "test-model", NULL), 0);
	stub_reset();

	void *params = rec_axis_decode(slot,
	                               "query='hello world' m=10 min_sim=0.2");
	ASSERT_NOT_NULL(params);
	ASSERT_EQ(stub_calls, 1);
	ASSERT(!strcmp(stub_text, "hello world"),
	       "multi-word text arrives verbatim");

	/* canned vector must drive fill+rank: stub at ref 7 scores 1.0 */
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(sepal_put(vs, 7, stub_vec, 3), 0);
	ASSERT_EQ(rec_axis_set_ctx(slot, vs), 0);
	const rec_axis_t *axis = rec_axis_get(slot);
	rec_set_t *via = rec_set_new();
	ASSERT_EQ(axis->fill(axis->ctx, params, via), 0);
	{
		size_t i;
		int found = 0;
		const rec_ref_t *refs = rec_set_at(via);
		for (i = 0; i < rec_set_count(via); i++)
			if (refs[i] == 7)
				found = 1;
		ASSERT(found, "query vector fills its own ref");
	}
	float score = -1.0f;
	ASSERT_EQ(axis->rank(axis->ctx, params, 7, &score), 0);
	ASSERT_NEAR(score, 1.0f, 1e-6f);
	rec_set_free(via);
	sepal_close(vs);
	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);
}

static void
test_decode_query_unquoted(void)
{
	printf("=== decode: bare query=beacon embeds as one token ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	ASSERT_EQ(sepal_configure_embeddings("http://localhost:9/none",
	                                     "test-model", NULL), 0);
	stub_reset();

	void *params = rec_axis_decode(slot, "query=beacon");
	ASSERT_NOT_NULL(params);
	ASSERT_EQ(stub_calls, 1);
	ASSERT(!strcmp(stub_text, "beacon"), "single token arrives verbatim");
	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);
}

static void
test_decode_query_beats_file(void)
{
	printf("=== decode: non-empty query wins over file= ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	ASSERT_EQ(sepal_configure_embeddings("http://localhost:9/none",
	                                     "test-model", NULL), 0);

	/* decoy file vector differs from the stub: {1,0,0} vs stub */
	float decoy[3] = { 1.0f, 0.0f, 0.0f };
	char path[] = "/tmp/test_sepal_query_file_XXXXXX";
	int fd = mkstemp(path);
	ASSERT(fd >= 0, "mkstemp ok");
	FILE *f = fdopen(fd, "wb");
	ASSERT_NOT_NULL(f);
	ASSERT_EQ(fwrite(decoy, sizeof(float), 3, f), 3);
	fclose(f);

	stub_reset();
	char params_str[512];
	snprintf(params_str, sizeof(params_str),
	         "query='hello' file=%s qdim=3 m=10", path);
	void *params = rec_axis_decode(slot, params_str);
	ASSERT_NOT_NULL(params);
	ASSERT_EQ(stub_calls, 1);
	ASSERT(!strcmp(stub_text, "hello"), "query text used, not file");

	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(sepal_put(vs, 7, stub_vec, 3), 0);
	ASSERT_EQ(sepal_put(vs, 5, decoy, 3), 0);
	ASSERT_EQ(rec_axis_set_ctx(slot, vs), 0);
	const rec_axis_t *axis = rec_axis_get(slot);
	float score = -1.0f;
	ASSERT_EQ(axis->rank(axis->ctx, params, 7, &score), 0);
	ASSERT_NEAR(score, 1.0f, 1e-6f);
	sepal_close(vs);
	remove(path);
	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);
}

static void
test_decode_query_empty_is_null(void)
{
	printf("=== decode: empty query + no file -> NULL ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	ASSERT_EQ(sepal_configure_embeddings("http://localhost:9/none",
	                                     "test-model", NULL), 0);
	stub_reset();

	ASSERT(rec_axis_decode(slot, "query= m=10") == NULL,
	       "decode: bare empty query -> NULL");
	ASSERT(rec_axis_decode(slot, "query='' m=10") == NULL,
	       "decode: quoted empty query -> NULL");
	ASSERT_EQ(stub_calls, 0);
	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);
}

static void
test_decode_query_unconfigured_is_null(void)
{
	printf("=== decode: unconfigured embedder -> NULL, fetch untouched ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);
	stub_reset();

	ASSERT(rec_axis_decode(slot, "query='hello' m=10") == NULL,
	       "decode: query with no embedder -> NULL");
	ASSERT_EQ(stub_calls, 0);
}

static void
test_decode_query_end_to_end(void)
{
	printf("=== decode: query= end-to-end (put floats, fill+rank find ref) ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	ASSERT_EQ(sepal_configure_embeddings("http://localhost:9/none",
	                                     "test-model", NULL), 0);
	stub_reset();

	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	float decoy[3] = { 1.0f, 0.0f, 0.0f };
	ASSERT_EQ(sepal_put(vs, 3, decoy, 3), 0);
	ASSERT_EQ(sepal_put(vs, 9, stub_vec, 3), 0);

	void *params = rec_axis_decode(slot, "query='hello' m=10 min_sim=0.9");
	ASSERT_NOT_NULL(params);
	ASSERT_EQ(rec_axis_set_ctx(slot, vs), 0);
	const rec_axis_t *axis = rec_axis_get(slot);

	rec_set_t *via = rec_set_new();
	ASSERT_EQ(axis->fill(axis->ctx, params, via), 0);
	{
		size_t i;
		int found = 0;
		const rec_ref_t *refs = rec_set_at(via);
		for (i = 0; i < rec_set_count(via); i++)
			if (refs[i] == 9)
				found = 1;
		ASSERT(found, "stored stub ref recovered via query= fill");
	}
	float score = -1.0f;
	ASSERT_EQ(axis->rank(axis->ctx, params, 9, &score), 0);
	ASSERT_NEAR(score, sepal_cosine(stub_vec, stub_vec, 3), 1e-6f);
	rec_set_free(via);
	sepal_close(vs);
	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);
}

int
main(void)
{
	test_decode_query_quoted();
	test_decode_query_unquoted();
	test_decode_query_beats_file();
	test_decode_query_empty_is_null();
	test_decode_query_unconfigured_is_null();
	test_decode_query_end_to_end();
	return test_summary();
}
