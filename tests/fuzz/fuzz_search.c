/* fuzz_search.c — standalone search/set fuzzer (deterministic, no libFuzzer).
 * Adversarial query dims and knobs; any crash or invariant violation is a bug. */

#include "../test_common.h"

#ifndef FUZZ_ITERS
#define FUZZ_ITERS 20000
#endif

int
main(void)
{
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	if (!vs)
		return 1;

	sepal_rng_t r = { 92 };
	float v[128];
	for (size_t i = 0; i < 64; i++) {
		rng_unit_vector(&r, v, 128);
		if (sepal_put(vs, (rec_ref_t)i, v, 128) != 0)
			return 2;
	}

	uint8_t qbuf[2048 * 4];
	sepal_hit_t out[64];
	unsigned search_ok = 0, fill_ok = 0;

	for (unsigned i = 0; i < FUZZ_ITERS; i++) {
		size_t qdim = 1 + (size_t)(rng_next(&r) % 2048);
		for (size_t j = 0; j < qdim; j++)
			((float *)qbuf)[j] = rng_unit(&r);
		size_t k = (size_t)(rng_next(&r) % 70);
		size_t m = (size_t)(rng_next(&r) % 300);
		float mins = (float)(rng_next(&r) % 200) / 100.0f - 1.0f;
		const float *q = (const float *)qbuf;

		size_t h = sepal_search(vs, q, qdim, k, mins, m, out);
		search_ok++;
		if (h > k || (m > 0 && h > m)) {
			printf("invariant broken: h=%zu k=%zu m=%zu\n", h, k, m);
			return 3;
		}
		for (size_t j = 1; j < h; j++)
			if (out[j - 1].score < out[j].score)
				return 4;   /* ordering violated */

		rec_set_t *s = rec_set_new();
		if (!s)
			return 5;
		int fr = sepal_fill_approx(vs, q, qdim, m, mins, s);
		if (fr == 0) {
			fill_ok++;
			if (rec_set_approx(s) != REC_SET_APPROX)
				return 6;
		}
		struct sepal_rank_ctx ctx = { vs, q, qdim, mins };
		float sc = 0.0f;
		(void)sepal_rank(&ctx, (rec_ref_t)(rng_next(&r) % 64), &sc);
		rec_set_free(s);
		(void)search_ok;
	}

	printf("fuzz_search: %u iters, search %u ok, fill %u ok\n",
	       FUZZ_ITERS, search_ok, fill_ok);
	sepal_close(vs);
	return 0;
}