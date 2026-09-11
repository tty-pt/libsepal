/* bench_search.c — exact brute-force scan+cosine vs two-stage ANN.
 *
 * Deterministic PRNG corpus N x {100, 1k, 10k} at full dims {384, 768},
 * unit vectors; query batch 100, k=10, m=100. Records (study §12.4):
 * N, dim, exact µs, prefilter-proxy µs (sepal_fill_approx), rerank-proxy µs
 * (per-ref sepal_rank), two-stage total µs, recall@10. */

#include "../test_common.h"

#include <sys/time.h>

static uint64_t
usec_now(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

static sepal_vecstore_t *
build(size_t n, size_t dim, sepal_rng_t *r)
{
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	if (!vs)
		return NULL;
	float *v = malloc(dim * sizeof(float));
	if (!v) {
		sepal_close(vs);
		return NULL;
	}
	for (size_t i = 0; i < n; i++) {
		rng_unit_vector(r, v, dim);
		sepal_put(vs, (rec_ref_t)i, v, dim);
	}
	free(v);
	return vs;
}

static float *
flat_base(sepal_vecstore_t *vs, size_t n, size_t dim)
{
	float *b = malloc(n * dim * sizeof(float));
	if (!b)
		return NULL;
	for (size_t i = 0; i < n; i++)
		sepal_get(vs, (rec_ref_t)i, b + i * dim, 4096);
	return b;
}

static void
run(size_t n, size_t dim, sepal_rng_t *r)
{
	const size_t K = 10, M = 100, QBATCH = 100;
	sepal_vecstore_t *vs = build(n, dim, r);
	if (!vs) {
		printf("build failed n=%zu dim=%zu\n", n, dim);
		return;
	}
	float *b = flat_base(vs, n, dim);
	if (!b) {
		sepal_close(vs);
		return;
	}
	float *qq = malloc(QBATCH * dim * sizeof(float));
	if (!qq) {
		free(b);
		sepal_close(vs);
		return;
	}
	for (size_t i = 0; i < QBATCH; i++)
		rng_unit_vector(r, qq + i * dim, dim);

	sepal_hit_t sh[16];
	sepal_bf_hit_t bf[16];
	rec_set_t *fillset = rec_set_new();
	struct sepal_rank_ctx ctx = { vs, NULL, dim, -2.0f };
	/* timers ignore loop scaffolding: batch all, divide after */
	uint64_t t0, t_exact = 0, t_fill = 0, t_rank = 0, t_search = 0;
	unsigned recalled = 0;
	double mean_recall = 0.0;

	for (size_t qi = 0; qi < QBATCH; qi++) {
		const float *q = qq + qi * dim;

		t0 = usec_now();
		size_t bn = brute_force_search(b, n, dim, 256, q, dim, K, -2.0f, bf);
		t_exact += usec_now() - t0;
		(void)bn;

		t0 = usec_now();
		sepal_fill_approx(vs, q, dim, M, -2.0f, fillset);
		t_fill += usec_now() - t0;
		const rec_ref_t *got = rec_set_at(fillset);
		size_t cnt = rec_set_count(fillset);

		t0 = usec_now();
		ctx.q = q;
		float sc;
		for (size_t j = 0; j < cnt && j < M; j++)
			(void)sepal_rank(&ctx, got[j], &sc);
		t_rank += usec_now() - t0;
		rec_set_free(fillset);
		fillset = rec_set_new();

		t0 = usec_now();
		size_t sn = sepal_search(vs, q, dim, K, -2.0f, M, sh);
		t_search += usec_now() - t0;

		int in_set[16] = { 0 };
		size_t denom = bn < K ? bn : K;
		for (size_t j = 0; j < bn && j < K; j++) {
			for (size_t t = 0; t < sn; t++)
				if (sh[t].ref == bf[j].ref) {
					in_set[j] = 1;
					break;
				}
		}
		if (denom > 0) {
			size_t got = 0;
			for (size_t j = 0; j < denom; j++)
				got += (size_t)in_set[j];
			mean_recall += (double)got / (double)denom;
			recalled += (got == denom);
		}
	}

	printf("%-7zu %-5zu %10llu %10llu %10llu %10llu   %.3f  %.3f\n",
	       n, dim,  (unsigned long long)(t_exact / QBATCH), (unsigned long long)(t_fill / QBATCH),
	       (unsigned long long)(t_rank / QBATCH), (unsigned long long)(t_search / QBATCH),
	       (double)recalled / (double)QBATCH,
	       mean_recall / (double)QBATCH);

	free(qq);
	free(b);
	rec_set_free(fillset);
	sepal_close(vs);
}

int
main(void)
{
	printf("%-7s %-5s %10s %10s %10s %10s   %s  %s\n",
	       "N", "dim", "exact_us", "pref_us", "rerank_us", "total_us",
	       "full@10", "mean_r@10");
	sepal_rng_t r = { 1001 };
	size_t ns[] = { 100, 1000, 10000 };
	size_t ds[] = { 384, 768 };
	for (size_t di = 0; di < 2; di++)
		for (size_t ni = 0; ni < 3; ni++)
			run(ns[ni], ds[di], &r);
	return 0;
}