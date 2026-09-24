/* twostage.c — the two-stage ANN pipeline made explicit.
 *
 * 1. sepal_fill_approx(): Hamming prefilter -> top-m candidate set
 *    (approximate; declares itself on the set).
 * 2. sepal_rank(): exact-cosine rerank of any ref, one at a time.
 * 3. sepal_search(): both stages in one call (equivalent output). */

#include <stdio.h>

#include <ttypt/sepal.h>

int
main(void)
{
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	if (!vs)
		return 1;

	float v[128];
	for (unsigned i = 0; i < 128; i++)
		v[i] = 0.0f;
	for (unsigned r = 0; r < 50; r++) {
		for (unsigned i = 0; i < 128; i++)
			v[i] = (float)(((r * 131 + i * 17) % 200) - 100) / 100.0f;
		sepal_put(vs, (rec_ref_t)r, v, 128);
	}

	float q[128];
	for (unsigned i = 0; i < 128; i++)
		q[i] = (float)(((i * 17) % 200) - 100) / 100.0f;

	/* stage 1: candidate set */
	rec_set_t *cands = rec_set_new();
	sepal_fill_approx(vs, q, 128, 10, -2.0f, cands);
	printf("candidates: %zu (approx=%d bound=%.2f)\n",
	       rec_set_count(cands), rec_set_approx(cands),
	       rec_set_recall_bound(cands));

	/* stage 2: rerank each candidate */
	struct sepal_rank_ctx ctx = { vs, q, 128, -2.0f };
	const rec_ref_t *refs = rec_set_at(cands);
	for (size_t i = 0; i < rec_set_count(cands); i++) {
		float sc;
		if (sepal_rank(&ctx, refs[i], &sc) == 0)
			printf("  ref=%u cosine=%.4f\n",
			       refs[i], sc);
	}
	rec_set_free(cands);

	/* one call, same answer */
	sepal_hit_t hits[5];
	size_t n = sepal_search(vs, q, 128, 5, -2.0f, 10, hits);
	printf("top-%zu via sepal_search:\n", n);
	for (size_t i = 0; i < n; i++)
		printf("  ref=%u cosine=%.4f\n",
		       hits[i].ref, hits[i].score);

	sepal_close(vs);
	return 0;
}
