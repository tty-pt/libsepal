/* kernel_form.c — recall-kernel form of a semantic query.
 *
 * Fill an approximate candidate set, intersect it with an exact set built
 * elsewhere (e.g. a token or geo prefilter), then rerank the survivors.
 * The join propagates the approximate flag, so downstream recall stays
 * honest about what it owes. */

#include <stdio.h>

#include <ttypt/sepal.h>

int
main(void)
{
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	if (!vs)
		return 1;

	float v[64];
	for (unsigned r = 0; r < 100; r++) {
		for (unsigned i = 0; i < 64; i++)
			v[i] = (float)(((r * 131 + i * 17) % 200) - 100) / 100.0f;
		sepal_put(vs, (rec_ref_t)r, v, 64);
	}

	float q[64];
	for (unsigned i = 0; i < 64; i++)
		q[i] = (float)(((i * 17) % 200) - 100) / 100.0f;

	/* an exact set from elsewhere: refs 0..49 */
	rec_set_t *exact = rec_set_new();
	for (unsigned r = 0; r < 50; r++)
		rec_set_push(exact, r);
	rec_set_seal(exact);

	/* approximate semantic candidates */
	rec_set_t *sem = rec_set_new();
	sepal_fill_approx(vs, q, 64, 20, -2.0f, sem);

	/* the join stays honest */
	rec_set_t *joined = rec_set_new();
	rec_set_intersect(joined, exact, sem);
	printf("exact=%zu approx=%zu joined=%zu approx=%d bound=%.3f\n",
	       rec_set_count(exact), rec_set_count(sem),
	       rec_set_count(joined), rec_set_approx(joined),
	       rec_set_recall_bound(joined));

	/* rerank the survivors exactly */
	struct sepal_rank_ctx ctx = { vs, q, 64, -2.0f };
	rec_rank_t *rk = rec_rank_new(5, -2.0f);
	const rec_ref_t *refs = rec_set_at(joined);
	for (size_t i = 0; i < rec_set_count(joined); i++) {
		float sc;
		if (sepal_rank(&ctx, refs[i], &sc) == 0)
			rec_rank_push(rk, refs[i], sc);
	}
	rec_ref_t out[5];
	float scs[5];
	size_t n = rec_rank_sorted(rk, out, scs);
	for (size_t i = 0; i < n; i++)
		printf("  ref=%u cosine=%.4f\n",
		       out[i], scs[i]);

	rec_rank_free(rk);
	rec_set_free(exact);
	rec_set_free(sem);
	rec_set_free(joined);
	sepal_close(vs);
	return 0;
}
