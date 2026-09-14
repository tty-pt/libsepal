/*
 * test_axis_store_live.c — 2A-1: env-gated live curl embed test.
 *
 * Skipped by default (exits 0 with a note). Runs ONLY when
 * SEPAL_LIVE_EMBED=1, using the endpoint supplied via
 * SEPAL_LIVE_EMBED_URL / SEPAL_LIVE_EMBED_MODEL / SEPAL_LIVE_EMBED_KEY.
 * Unlike test_axis_store.c this file does NOT override sepal_embed_fetch,
 * so the real (weak, curl) implementation runs.
 */

#include "../test_common.h"
#include <ttypt/rec.h>

#include <errno.h>

static void
config_from_env(const char **url, const char **model, const char **key)
{
	*url = getenv("SEPAL_LIVE_EMBED_URL");
	*model = getenv("SEPAL_LIVE_EMBED_MODEL");
	if (!*model)
		*model = "test-model";
	*key = getenv("SEPAL_LIVE_EMBED_KEY");
}

int
main(void)
{
	const char *live = getenv("SEPAL_LIVE_EMBED");
	const char *url, *model, *key;

	if (!live || !*live || !strcmp(live, "0")) {
		printf("SKIP: live curl embed (set SEPAL_LIVE_EMBED=1 to run)\n");
		return 0;
	}
	config_from_env(&url, &model, &key);
	if (!url || !*url) {
		printf("SKIP: SEPAL_LIVE_EMBED_URL not set\n");
		return 0;
	}

	printf("=== store: live curl embed of a short string ===\n");
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	ASSERT_NOT_NULL(vs);
	ASSERT_EQ(sepal_configure_embeddings(url, model, key), 0);

	errno = 0;
	ASSERT_EQ(rec_axis_store(vs, NULL, 1, "the quick brown fox"), 0);
	ASSERT(sepal_dim(vs, 1) > 0, "live embed stored nonzero dims");

	{
		float out[2048];
		size_t got = sepal_get(vs, 1, out, 2048);
		ASSERT(got == sepal_dim(vs, 1), "live embed retrievable");
		ASSERT(got <= 2048, "live embed within SEPAL_VEC_MAX");
	}

	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);
	sepal_close(vs);
	return test_summary();
}
