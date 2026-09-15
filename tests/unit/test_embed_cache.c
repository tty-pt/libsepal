/*
 * test_embed_cache.c — L2 query-embed cache (AXIS-EFF plan): the on-disk
 * (model, text) cache around sepal_embed_fetch. Offline via the same
 * canned-vector stub as test_axis_decode_query.c; the cache dir comes from
 * the QMAP_SEPAL_EMBED_CACHE_DIR env override so this test never depends
 * on rec_axis_open path parsing.
 *
 * Behaviors pinned:
 *   - miss → live fetch (stub counted); 2nd identical query → cache hit,
 *     fetch NOT called (call count unchanged).
 *   - different text → miss again; different model → miss again.
 *   - QMAP_SEPAL_EMBED_CACHE=0 → cache bypassed (every decode fetches).
 *   - unconfigured embedder → NULL, cache + fetch untouched.
 *   - corrupt cache file → graceful miss (fetch each time, never crash),
 *     and a garbage prefix keeps later valid records unreachable (scanner
 *     bails at the first malformed entry) — best-effort is documented.
 *   - oversized cache file (> SE_EMBED_CACHE_MAX) → skipped entirely:
 *     decode misses and the append is refused (bounded growth).
 */

#include "../test_common.h"
#include <ttypt/rec.h>
#include <unistd.h>

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

/* Cache file lives at <dir>/sepal-embed-cache.bin (same derivation the
 * library uses when cache_dir is the env override). */
static char cache_dir[64];
static char cache_path[96];

static void
cache_on(void)
{
	ASSERT_EQ(setenv("QMAP_SEPAL_EMBED_CACHE_DIR", cache_dir, 1), 0);
	ASSERT_EQ(unsetenv("QMAP_SEPAL_EMBED_CACHE"), 0);
	/* cache_dir only lands in the cfg when rec_axis_env_config() runs, so
	 * re-apply after the env is in place. */
	ASSERT_EQ(rec_axis_env_config(), 0);
}

static void
cache_off(void)
{
	ASSERT_EQ(setenv("QMAP_SEPAL_EMBED_CACHE", "0", 1), 0);
}

/* cache_dir is populated only by rec_axis_env_config / rec_axis_open —
 * sepal_configure_embeddings() itself never touches it. Configure through
 * the public env seam: clear first (so a stale cfg can't linger), set the
 * env pair the library actually reads, then apply. */
static void
config_embed(const char *model)
{
	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);
	ASSERT_EQ(setenv("QMAP_SEPAL_EMBED_URL", "http://localhost:9/none", 1), 0);
	if (model)
		ASSERT_EQ(setenv("QMAP_SEPAL_EMBED_MODEL", model, 1), 0);
	else
		ASSERT_EQ(unsetenv("QMAP_SEPAL_EMBED_MODEL"), 0);
	ASSERT_EQ(rec_axis_env_config(), 0);
}

static void
config_unconfigured(void)
{
	ASSERT_EQ(sepal_configure_embeddings(NULL, NULL, NULL), 0);
	ASSERT_EQ(unsetenv("QMAP_SEPAL_EMBED_URL"), 0);
	ASSERT_EQ(unsetenv("QMAP_SEPAL_EMBED_MODEL"), 0);
	ASSERT_EQ(rec_axis_env_config(), 0);
}

static void
write_garbage(size_t n)
{
	FILE *f = fopen(cache_path, "wb");
	ASSERT_NOT_NULL(f);
	unsigned char blob[4096];
	memset(blob, 0xA5, sizeof(blob));
	while (n > 0) {
		size_t w = n < sizeof(blob) ? n : sizeof(blob);
		ASSERT_EQ(fwrite(blob, 1, w, f), w);
		n -= w;
	}
	fclose(f);
}

static void
test_cache_miss_then_hit(void)
{
	printf("=== cache: miss, then repeat query hits without fetch ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	config_embed("cache-model");
	cache_on();
	remove(cache_path);
	stub_reset();

	void *p1 = rec_axis_decode(slot, "query='harbor lights' m=10");
	ASSERT_NOT_NULL(p1);
	ASSERT_EQ(stub_calls, 1);
	ASSERT(!strcmp(stub_text, "harbor lights"),
	       "first query embeds the live path");

	void *p2 = rec_axis_decode(slot, "query='harbor lights' m=10");
	ASSERT_NOT_NULL(p2);
	ASSERT(stub_calls == 1, "identical query served from the cache");
	config_unconfigured();
}

static void
test_cache_new_text_and_model_miss(void)
{
	printf("=== cache: new text and new model are misses ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	config_embed("cache-model");
	cache_on();
	remove(cache_path);
	stub_reset();

	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='alpha'"));
	ASSERT_EQ(stub_calls, 1);
	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='beta'"));
	ASSERT(stub_calls == 2, "different text → live fetch");

	config_embed("cache-model-2");
	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='alpha'"));
	ASSERT(stub_calls == 3, "same text, new model → live fetch");
	config_unconfigured();
}

static void
test_cache_env_optout(void)
{
	printf("=== cache: QMAP_SEPAL_EMBED_CACHE=0 bypasses the cache ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	config_embed("cache-model");
	cache_on();
	remove(cache_path);
	stub_reset();
	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='alpha'"));
	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='alpha'"));
	ASSERT(stub_calls == 1, "cache hit under opt-in");

	cache_off();
	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='alpha'"));
	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='alpha'"));
	ASSERT(stub_calls == 3, "opt-out: every decode fetches");
	config_unconfigured();
}

static void
test_cache_unconfigured(void)
{
	printf("=== cache: unconfigured embedder → NULL, nothing touched ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	config_unconfigured();
	cache_on();
	remove(cache_path);
	stub_reset();

	ASSERT(rec_axis_decode(slot, "query='alpha'") == NULL,
	       "no embedder → NULL");
	ASSERT(stub_calls == 0, "no embedder → no fetch, no cache write");
}

static void
test_cache_corrupt_file(void)
{
	printf("=== cache: corrupt file → graceful miss, never crashes ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	config_embed("cache-model");
	cache_on();
	stub_reset();

	write_garbage(64);
	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='alpha'"));
	ASSERT(stub_calls == 1, "garbage prefix → miss");
	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='alpha'"));
	ASSERT(stub_calls == 2, "scanner bails at malformed entry → miss");
	config_unconfigured();
}

static void
test_cache_oversized(void)
{
	printf("=== cache: oversized file → skipped, growth refused ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	config_embed("cache-model");
	cache_on();
	stub_reset();

	write_garbage((8u << 20) + 4096);
	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='alpha'"));
	ASSERT(stub_calls == 1, "oversized → miss");
	ASSERT_NOT_NULL(rec_axis_decode(slot, "query='alpha'"));
	ASSERT(stub_calls == 2, "append refused past cap → miss again");
	config_unconfigured();
}

int
main(void)
{
	char tmpl[] = "/tmp/test_sepal_embed_cache_XXXXXX";
	char *dir = mkdtemp(tmpl);
	ASSERT(dir != NULL, "mkdtemp ok");
	snprintf(cache_dir, sizeof(cache_dir), "%s", dir);
	snprintf(cache_path, sizeof(cache_path), "%s/sepal-embed-cache.bin",
	         cache_dir);

	test_cache_miss_then_hit();
	test_cache_new_text_and_model_miss();
	test_cache_env_optout();
	test_cache_unconfigured();
	test_cache_corrupt_file();
	test_cache_oversized();

	remove(cache_path);
	rmdir(cache_dir);
	return test_summary();
}