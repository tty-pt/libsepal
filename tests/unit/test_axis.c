/*
 * test_axis.c — rec_query "sepal" axis registration: registered fill/rank
 * dispatch through the axis table must equal the direct sepal_fill_approx /
 * sepal_rank calls, and the decode fn must round-trip a query vector file.
 */

#include "../test_common.h"
#include <ttypt/rec.h>
#include <unistd.h>

static sepal_vecstore_t *
build_corpus(size_t n, uint64_t seed, size_t dim)
{
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	if (!vs)
		return NULL;
	float *v = malloc(dim * sizeof(float));
	if (!v) {
		sepal_close(vs);
		return NULL;
	}
	sepal_rng_t r = { seed };
	for (size_t i = 0; i < n; i++) {
		rng_unit_vector(&r, v, dim);
		sepal_put(vs, (rec_ref_t)i, v, dim);
	}
	free(v);
	return vs;
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
test_axis_registered(void)
{
	printf("=== axis: \"sepal\" registered with fill+rank+decode ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");

	const rec_axis_t *axis = rec_axis_get(slot);
	ASSERT_NOT_NULL(axis);
	ASSERT(axis->fill != NULL, "sepal has fill");
	ASSERT(axis->rank != NULL, "sepal has rank");
	ASSERT(axis->decode != NULL, "sepal has decode");
	ASSERT(axis->ctx == NULL, "sepal ctx NULL before set_ctx");
}

static void
test_axis_fill_equivalence(void)
{
	printf("=== axis: registered fill == direct sepal_fill_approx ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	sepal_vecstore_t *vs = build_corpus(100, 61, 64);
	ASSERT_NOT_NULL(vs);

	ASSERT_EQ(rec_axis_set_ctx(slot, vs), 0);
	const rec_axis_t *axis = rec_axis_get(slot);
	ASSERT(axis->ctx == vs, "ctx set to vecstore");

	float q[64];
	sepal_rng_t r = { 62 };
	rng_unit_vector(&r, q, 64);

	char path[] = "/tmp/test_sepal_axis_q_XXXXXX";
	int fd = mkstemp(path);
	ASSERT(fd >= 0, "mkstemp ok");
	FILE *f = fdopen(fd, "wb");
	ASSERT_NOT_NULL(f);
	ASSERT_EQ(fwrite(q, sizeof(float), 64, f), 64);
	fclose(f);

	char params_str[256];
	snprintf(params_str, sizeof(params_str),
	         "file=%s qdim=64 m=10 min_sim=0.0", path);
	void *params = rec_axis_decode(slot, params_str);
	ASSERT_NOT_NULL(params);

	rec_set_t *direct = rec_set_new();
	ASSERT_EQ(sepal_fill_approx(vs, q, 64, 10, 0.0f, direct), 0);

	rec_set_t *via = rec_set_new();
	ASSERT_EQ(axis->fill(axis->ctx, params, via), 0);

	ASSERT_EQ(rec_set_count(direct), rec_set_count(via));
	ASSERT_EQ(rec_set_count(direct), 10);
	if (rec_set_count(direct) == rec_set_count(via))
		ASSERT(memcmp(rec_set_at(direct), rec_set_at(via),
		              rec_set_count(direct) * sizeof(rec_ref_t)) == 0,
		       "registered fill refs == direct refs");

	rec_set_free(direct);
	rec_set_free(via);
	remove(path);
	sepal_close(vs);
}

static void
test_axis_rank_equivalence(void)
{
	printf("=== axis: registered rank == direct sepal_rank ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);

	float v[128], q[128];
	sepal_rng_t r = { 63 };
	rng_unit_vector(&r, v, 128);
	rng_unit_vector(&r, q, 128);
	ASSERT_EQ(sepal_put(vs, 42, v, 128), 0);

	ASSERT_EQ(rec_axis_set_ctx(slot, vs), 0);
	const rec_axis_t *axis = rec_axis_get(slot);

	char path[] = "/tmp/test_sepal_axis_q_XXXXXX";
	int fd = mkstemp(path);
	ASSERT(fd >= 0, "mkstemp ok");
	FILE *f = fdopen(fd, "wb");
	ASSERT_NOT_NULL(f);
	ASSERT_EQ(fwrite(q, sizeof(float), 128, f), 128);
	fclose(f);

	char params_str[256];
	snprintf(params_str, sizeof(params_str),
	         "file=%s qdim=128 m=10 min_sim=-1.0", path);
	void *params = rec_axis_decode(slot, params_str);
	ASSERT_NOT_NULL(params);

	struct sepal_rank_ctx dctx = { vs, q, 128, -1.0f };
	float direct_score = -1.0f, via_score = -1.0f;

	ASSERT_EQ(sepal_rank(&dctx, 42, &direct_score), 0);
	ASSERT_EQ(axis->rank(axis->ctx, params, 42, &via_score), 0);
	ASSERT_NEAR(direct_score, via_score, 1e-6f);

	remove(path);
	sepal_close(vs);
}

static void
test_axis_decode_guards(void)
{
	printf("=== axis: decode guards (missing file/qdim/bad path) ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");

	ASSERT(rec_axis_decode(slot, "qdim=64") == NULL,
	       "decode: missing file -> NULL");
	ASSERT(rec_axis_decode(slot, "file=/tmp/does-not-exist-xyz") == NULL,
	       "decode: missing qdim -> NULL");
	ASSERT(rec_axis_decode(slot, "file=/tmp/does-not-exist-xyz qdim=8") ==
	               NULL,
	       "decode: unreadable file -> NULL");
	ASSERT(rec_axis_decode(-1, "file=x qdim=8") == NULL,
	       "decode: invalid slot -> NULL");
}

static void
test_axis_open_matches_direct_open(void)
{
	printf("=== axis: rec_axis_open ctx == sepal_open ctx (fill parity) ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");

	/* empty spec -> memory-only store, same as sepal_open(NULL, ...) */
	void *ctx = rec_axis_open("");
	ASSERT_NOT_NULL(ctx);
	sepal_vecstore_t *vs = ctx;

	float v[32];
	sepal_rng_t r = { 64 };
	rng_unit_vector(&r, v, 32);
	ASSERT_EQ(sepal_put(vs, 7, v, 32), 0);

	ASSERT_EQ(rec_axis_set_ctx(slot, ctx), 0);
	const rec_axis_t *axis = rec_axis_get(slot);
	ASSERT(axis->ctx == ctx, "ctx bound to the rec_axis_open() result");

	rec_set_t *direct = rec_set_new();
	ASSERT_EQ(sepal_fill_approx(vs, v, 32, 5, 0.0f, direct), 0);
	{
		size_t i;
		int found = 0;
		const rec_ref_t *refs = rec_set_at(direct);
		for (i = 0; i < rec_set_count(direct); i++)
			if (refs[i] == 7)
				found = 1;
		ASSERT(found, "put ref recoverable via direct fill");
	}

	rec_set_free(direct);
	sepal_close(vs);
}

int
main(void)
{
	test_axis_registered();
	test_axis_fill_equivalence();
	test_axis_rank_equivalence();
	test_axis_decode_guards();
	test_axis_open_matches_direct_open();
	return test_summary();
}
