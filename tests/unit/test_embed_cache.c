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

/* ── D14 axis-contributed CLI options (external dilemma: the qmap CLI
 *    broadcasts --query / --min-sim to every bound axis declaring them).
 *    The convention symbols are dlsym'd by qmap; the tests call them
 *    directly through the shared lib, exactly like rec_axis_env_config. ── */

extern int rec_axis_config_arg(const char *name, const char *value);
extern const struct rec_axis_cli_read {
	const char *name;
	int has_arg;
	const char *help;
} *rec_axis_cli_options(void);

/* mirror of the decode params layout (libsepal.c rec_sepal_params) */
struct cli_test_params {
	float *q;
	size_t qdim;
	size_t m;
	float min_sim;
};

static void
test_cli_options(void)
{
	printf("=== cli options: bare-leaf query fallback + spec>CLI merge ===\n");
	int slot = find_sepal_slot();
	ASSERT(slot >= 0, "sepal axis registered");
	config_embed("cli-model");
	cache_on();
	remove(cache_path);
	stub_reset();

	/* no CLI state yet → NULL spec stays the hard-NULL regression */
	ASSERT(rec_axis_get(slot)->decode(NULL) == NULL,
	       "bare NULL spec without --query stays NULL");

	/* declared surface advertises exactly query + min-sim */
	{
		const struct rec_axis_cli_read *o = rec_axis_cli_options();
		int n = 0;
		while (o && o[n].name)
			n++;
		ASSERT(n == 2, "cli table: query + min-sim");
		ASSERT(!strcmp(o[0].name, "query"), "first option is query");
		ASSERT(o[0].has_arg == 1, "query takes a value");
		ASSERT(!strcmp(o[1].name, "min-sim"), "second option is min-sim");
	}

	/* bare-leaf CLI fallback: --query + NULL spec embeds the CLI text.
	 * The expression evaluator calls axis->decode(n->value) directly
	 * (qmap.c), so a bare `sepal` leaf reaches the plugin as NULL (the
	 * rec_axis_decode() wrapper rejects NULL specs by contract). */
	ASSERT_EQ(rec_axis_config_arg("query", "south pier lamp"), 0);
	void *p = rec_axis_get(slot)->decode(NULL);
	ASSERT_NOT_NULL(p);
	ASSERT_EQ(stub_calls, 1);
	ASSERT(!strcmp(stub_text, "south pier lamp"),
	       "bare NULL spec uses --query text");

	/* an empty spec with --query rides the same path (may be a cache
	 * hit from the text just embedded above — params, not fetch, prove it) */
	stub_reset();
	p = rec_axis_decode(slot, "");
	ASSERT_NOT_NULL(p);
	{
		struct cli_test_params *tp = p;
		ASSERT_EQ(tp->qdim, (size_t)3);
		ASSERT(tp->q != NULL, "query vector present (--query ride)");
	}

	/* a leaf query wins over --query */
	stub_reset();
	p = rec_axis_decode(slot, "query='leaftext'");
	ASSERT_NOT_NULL(p);
	ASSERT(!strcmp(stub_text, "leaftext"), "leaf query beats --query");

	/* --min-sim merges into params when the leaf omits min_sim= */
	ASSERT_EQ(rec_axis_config_arg("min-sim", "0.5"), 0);
	stub_reset();
	p = rec_axis_decode(slot, "query='hello'");
	ASSERT_NOT_NULL(p);
	{
		struct cli_test_params *tp = p;
		ASSERT(tp->min_sim == 0.5f, "--min-sim applied from CLI");
		ASSERT_EQ(tp->qdim, (size_t)3);
		ASSERT(tp->q != NULL, "query vector present");
	}

	/* a leaf min_sim= beats --min-sim */
	p = rec_axis_decode(slot, "query='hello' min_sim=0.7");
	ASSERT_NOT_NULL(p);
	{
		struct cli_test_params *tp = p;
		ASSERT(tp->min_sim == 0.7f, "leaf min_sim beats --min-sim");
	}
	/* ... but reverting to a leaf without min_sim falls back again */
	p = rec_axis_decode(slot, "query='hello'");
	ASSERT_NOT_NULL(p);
	{
		struct cli_test_params *tp = p;
		ASSERT(tp->min_sim == 0.5f, "leaf omission falls back to --min-sim");
	}

	/* arg validation: NULL/NaN/out-of-range/unknown all rejected */
	ASSERT(rec_axis_config_arg("query", NULL) != 0, "query NULL rejected");
	ASSERT(rec_axis_config_arg("min-sim", NULL) != 0, "min-sim NULL rejected");
	ASSERT(rec_axis_config_arg("min-sim", "abc") != 0, "min-sim NaN rejected");
	ASSERT(rec_axis_config_arg("min-sim", "1.5") != 0, "min-sim out of range");
	ASSERT(rec_axis_config_arg("min-sim", "0.25") == 0, "min-sim in range ok");
	ASSERT(rec_axis_config_arg("bogus", "x") != 0, "unknown option rejected");
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
	test_cli_options();

	remove(cache_path);
	rmdir(cache_dir);
	return test_summary();
}