/*
 * libsepal.c — libsepal, the meaning (semantic) axis.
 *
 * Dim-tagged vector store + two-stage ANN search (Hamming prefilter over a
 * sign-bit sketch, exact-cosine rerank) + recall-kernel adapters.
 * See include/ttypt/sepal.h for the design contract and VEC1 blob format.
 * Persistence follows the mm invariant: sepal_close() calls qmap_save();
 * qmap_close() is never used (it would truncate the file to 0).
 */

#include "../include/ttypt/sepal.h"

#include <ttypt/qmap.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------------------.
 * VEC1 blob (also see sepal.h): 16 + 8W + 4*dim bytes, little-endian.      */

#define BLOB_MAGIC0 'V'
#define BLOB_MAGIC1 'E'
#define BLOB_MAGIC2 'C'
#define BLOB_MAGIC3 '1'
#define BLOB_VERSION 1
#define BLOB_HDR_LEN 16

static uint16_t
be16_r(const uint8_t *b)
{
	uint16_t v;
	memcpy(&v, b, sizeof(v));
	return v;
}

static void
be16_w(uint8_t *b, uint16_t v)
{
	memcpy(b, &v, sizeof(v));
}

static size_t
blob_calc_len(uint16_t dim, uint16_t sketch_words)
{
	return BLOB_HDR_LEN + (size_t)8 * sketch_words + (size_t)4 * dim;
}

static int
blob_field_valid(uint16_t dim, uint16_t full_dim, uint16_t sketch_words)
{
	if (dim == 0 || dim > SEPAL_EXACT_DIM || dim > full_dim)
		return 0;
	if (full_dim == 0 || full_dim > SEPAL_VEC_MAX)
		return 0;
	if (sketch_words != (uint16_t)((full_dim + 63) / 64))
		return 0;
	return 1;
}

static int
blob_parse(const uint8_t *b, size_t avail, sepal_blob_hdr_t *hdr)
{
	if (!b || avail < BLOB_HDR_LEN)
		return -1;
	if (b[0] != BLOB_MAGIC0 || b[1] != BLOB_MAGIC1 ||
	    b[2] != BLOB_MAGIC2 || b[3] != BLOB_MAGIC3)
		return -1;
	uint16_t ver  = be16_r(b + 4);
	uint16_t dim  = be16_r(b + 6);
	uint16_t full = be16_r(b + 8);
	uint16_t w    = be16_r(b + 10);
	if (ver != BLOB_VERSION || !blob_field_valid(dim, full, w))
		return -1;
	size_t len = blob_calc_len(dim, w);
	if (len > avail)
		return -1;
	if (hdr) {
		hdr->dim          = dim;
		hdr->full_dim     = full;
		hdr->sketch_words = w;
		memcpy(&hdr->norm, b + 12, sizeof(float));
	}
	return 0;
}

static const float *
blob_floats(const uint8_t *b)
{
	return (const float *)(const void *)(b + BLOB_HDR_LEN + 8 * be16_r(b + 10));
}

size_t
sepal_blob_len(const uint8_t *blob, size_t avail)
{
	sepal_blob_hdr_t h;
	if (blob_parse(blob, avail, &h) != 0)
		return 0;
	return blob_calc_len(h.dim, h.sketch_words);
}

int
sepal_blob_hdr(const uint8_t *blob, size_t avail, sepal_blob_hdr_t *hdr)
{
	if (!hdr)
		return -1;
	return blob_parse(blob, avail, hdr);
}

size_t
sepal_blob_put(uint8_t *buf, size_t avail, const float *v,
               size_t full_dim, size_t dim)
{
	if (!buf || !v)
		return 0;
	if (full_dim < 1 || full_dim > SEPAL_VEC_MAX)
		return 0;
	if (dim < 1 || dim > SEPAL_EXACT_DIM || dim > full_dim)
		return 0;
	size_t w = (full_dim + 63) / 64;
	size_t len = blob_calc_len((uint16_t)dim, (uint16_t)w);
	if (len > avail)
		return 0;

	buf[0] = BLOB_MAGIC0; buf[1] = BLOB_MAGIC1;
	buf[2] = BLOB_MAGIC2; buf[3] = BLOB_MAGIC3;
	be16_w(buf + 4, BLOB_VERSION);
	be16_w(buf + 6, (uint16_t)dim);
	be16_w(buf + 8, (uint16_t)full_dim);
	be16_w(buf + 10, (uint16_t)w);

	double s = 0.0;
	for (size_t i = 0; i < dim; i++)
		s += (double)v[i] * (double)v[i];
	float norm = (float)sqrt(s);
	memcpy(buf + 12, &norm, sizeof(norm));

	uint64_t *sk = (uint64_t *)(void *)(buf + BLOB_HDR_LEN);
	for (size_t wi = 0; wi < w; wi++) {
		uint64_t word = 0;
		size_t base = wi * 64;
		size_t hi = base + 64 < full_dim ? base + 64 : full_dim;
		for (size_t i = base; i < hi; i++)
			if (__builtin_signbit(v[i]))
				word |= 1ULL << (i - base);
		sk[wi] = word;
	}

	float *fs = (float *)(void *)(buf + BLOB_HDR_LEN + 8 * w);
	for (size_t i = 0; i < dim; i++)
		fs[i] = v[i];
	return len;
}

/* ----------------------------------------------------------------------.
 * Store                                                                */

struct sepal_vecstore {
	uint32_t hd;
	size_t   n;
};

static uint32_t sepal_kt = QM_MISS;
static uint32_t sepal_vt = QM_MISS;

static size_t
blob_measure(const void *data)
{
	const uint8_t *b = data;
	if (blob_parse(b, (size_t)1 << 30, NULL) != 0)
		return 0;
	return blob_calc_len(be16_r(b + 6), be16_r(b + 10));
}

static void
sepal_register_types(void)
{
	if (sepal_kt != QM_MISS)
		return;
	sepal_kt = qmap_reg(sizeof(rec_ref_t));
	sepal_vt = qmap_mreg(blob_measure);
}

sepal_vecstore_t *
sepal_open(const char *fname, int *err)
{
	int e = 0;
	sepal_register_types();

	sepal_vecstore_t *vs = calloc(1, sizeof(*vs));
	if (!vs) {
		e = -2;
		goto out;
	}
	uint32_t hd = qmap_open(fname, NULL, sepal_kt, sepal_vt, 0xFF, 0);
	if (hd == QM_MISS) {
		free(vs);
		vs = NULL;
		e = -1;
		goto out;
	}
vs->hd = hd;

/* an existing file already holds entries: restore the count */
{
	uint32_t cur = qmap_iter(hd, NULL, 0);
	const void *k, *v;
	while (qmap_next(&k, &v, cur))
		vs->n++;
	qmap_fin(cur);
}
out:
	if (err)
		*err = e;
	return vs;
}

void
sepal_close(sepal_vecstore_t *vs)
{
	if (!vs)
		return;
	qmap_save();   /* persist; never qmap_close (truncates the file to 0) */
	free(vs);
}

int
sepal_put(sepal_vecstore_t *vs, rec_ref_t ref, const float *v, size_t full_dim)
{
	if (!vs || !v || full_dim < 1 || full_dim > SEPAL_VEC_MAX)
		return -1;
	size_t dim = full_dim < SEPAL_EXACT_DIM ? full_dim : SEPAL_EXACT_DIM;
	size_t w = (full_dim + 63) / 64;
	size_t len = blob_calc_len((uint16_t)dim, (uint16_t)w);

	uint8_t *buf = malloc(len);
	if (!buf)
		return -1;
	if (sepal_blob_put(buf, len, v, full_dim, dim) != len) {
		free(buf);
		return -1;
	}
	int was_present = qmap_count(vs->hd, &ref) > 0;
	if (was_present)
		qmap_del(vs->hd, &ref);
	/* NOTE: qmap_put's position return is not inspected — mm ignores it
	 * too, since a put can land on position 0 legitimately. Presence is
	 * verified below instead. */
	qmap_put(vs->hd, &ref, buf);
	free(buf);
	if (qmap_count(vs->hd, &ref) == 0)
		return -1;
	if (!was_present)
		vs->n++;
	return 0;
}

int
sepal_del(sepal_vecstore_t *vs, rec_ref_t ref)
{
	if (!vs)
		return -1;
	if (qmap_count(vs->hd, &ref) == 0)
		return -1;
	qmap_del(vs->hd, &ref);
	vs->n--;
	return 0;
}

static const uint8_t *
lookup_blob(sepal_vecstore_t *vs, rec_ref_t ref, sepal_blob_hdr_t *hdr)
{
	if (!vs)
		return NULL;
	const void *b = qmap_get(vs->hd, &ref);
	if (!b)
		return NULL;
	if (blob_parse(b, (size_t)1 << 30, hdr) != 0)
		return NULL;
	return b;
}

size_t
sepal_get(sepal_vecstore_t *vs, rec_ref_t ref, float *out, size_t max)
{
	sepal_blob_hdr_t h;
	const uint8_t *b = lookup_blob(vs, ref, &h);
	if (!b || !out)
		return 0;
	if (max < h.dim)          /* mismatch — never truncate */
		return 0;
	memcpy(out, blob_floats(b), h.dim * sizeof(float));
	return h.dim;
}

size_t
sepal_dim(sepal_vecstore_t *vs, rec_ref_t ref)
{
	sepal_blob_hdr_t h;
	return lookup_blob(vs, ref, &h) ? h.dim : 0;
}

size_t
sepal_full_dim(sepal_vecstore_t *vs, rec_ref_t ref)
{
	sepal_blob_hdr_t h;
	return lookup_blob(vs, ref, &h) ? h.full_dim : 0;
}

size_t
sepal_n(const sepal_vecstore_t *vs)
{
	return vs ? vs->n : 0;
}

/* ----------------------------------------------------------------------.
 * Math                                                                */

float
sepal_cosine(const float *a, const float *b, size_t n)
{
	double dot = 0.0, na = 0.0, nb = 0.0;
	for (size_t i = 0; i < n; i++) {
		dot += (double)a[i] * (double)b[i];
		na += (double)a[i] * (double)a[i];
		nb += (double)b[i] * (double)b[i];
	}
	if (na <= 0.0 || nb <= 0.0)
		return 0.0f;
	return (float)(dot / sqrt(na * nb));
}

int
sepal_sketch(const float *v, size_t n, uint64_t *words, size_t nwords)
{
	if (!v || !words || n < 1 || n > SEPAL_VEC_MAX)
		return -1;
	size_t need = (n + 63) / 64;
	if (nwords < need)
		return -1;
	for (size_t wi = 0; wi < need; wi++) {
		uint64_t word = 0;
		size_t base = wi * 64;
		size_t hi = base + 64 < n ? base + 64 : n;
		for (size_t i = base; i < hi; i++)
			if (__builtin_signbit(v[i]))
				word |= 1ULL << (i - base);
		words[wi] = word;
	}
	return 0;
}

/* ----------------------------------------------------------------------.
 * Two-stage search                                                       */

/* Max-heap of the m best (smallest-hamming) candidates; root = worst. */
typedef struct {
	uint32_t h;
	rec_ref_t r;
} cand_t;

typedef struct {
	cand_t *a;
	size_t n, m;
} cand_heap_t;

static void
heap_sift_down(cand_heap_t *hp)
{
	size_t i = 0;
	for (;;) {
		size_t l = 2 * i + 1, r = 2 * i + 2, big = i;
		if (l < hp->n && hp->a[l].h > hp->a[big].h)
			big = l;
		if (r < hp->n && hp->a[r].h > hp->a[big].h)
			big = r;
		if (big == i)
			break;
		cand_t t = hp->a[i]; hp->a[i] = hp->a[big]; hp->a[big] = t;
		i = big;
	}
}

static void
heap_push(cand_heap_t *hp, uint32_t h, rec_ref_t r)
{
	if (hp->n < hp->m) {
		hp->a[hp->n].h = h;
		hp->a[hp->n].r = r;
		size_t i = hp->n++;
		while (i > 0) {
			size_t p = (i - 1) / 2;
			if (hp->a[p].h >= hp->a[i].h)
				break;
			cand_t t = hp->a[p]; hp->a[p] = hp->a[i]; hp->a[i] = t;
			i = p;
		}
	} else if (hp->n == hp->m && h < hp->a[0].h) {
		hp->a[0].h = h;
		hp->a[0].r = r;
		heap_sift_down(hp);
	}
}

/* Scan the store's blobs, keeping the m smallest Hamming refs. Returns 0 ok,
 * -1 on allocation failure. */
static int
prefilter(sepal_vecstore_t *vs, const float *q, size_t qdim,
          cand_heap_t *hp)
{
	uint64_t qw[SEPAL_VEC_MAX / 64];
	size_t wq = (qdim + 63) / 64;
	if (sepal_sketch(q, qdim, qw, wq) != 0)
		return -1;

	uint32_t cur = qmap_iter(vs->hd, NULL, 0);
	const void *k, *v;
	while (qmap_next(&k, &v, cur)) {
		sepal_blob_hdr_t h;
		if (blob_parse(v, (size_t)1 << 30, &h) != 0)
			continue;
		const uint64_t *sk = (const uint64_t *)(const void *)
			((const uint8_t *)v + BLOB_HDR_LEN);
		size_t words = wq < h.sketch_words ? wq : h.sketch_words;
		uint64_t hd = 0;
		for (size_t i = 0; i < words; i++)
			hd += (uint64_t)__builtin_popcountll(qw[i] ^ sk[i]);
		rec_ref_t ref;
		memcpy(&ref, k, sizeof(ref));
		heap_push(hp, (uint32_t)hd, ref);
	}
	qmap_fin(cur);
	return 0;
}

size_t
sepal_search(sepal_vecstore_t *vs, const float *q, size_t qdim,
             size_t k, float min_sim, size_t m, sepal_hit_t *out)
{
	if (!vs || !q || !out || k == 0)
		return 0;
	if (qdim < 1 || qdim > SEPAL_VEC_MAX)
		return 0;
	if (m == 0)
		m = SEPAL_M_DEFAULT * k;
	if (m > vs->n)
		m = vs->n;
	if (m == 0)
		return 0;

	cand_heap_t hp = { 0 };
	hp.m = m;
	hp.a = malloc(m * sizeof(cand_t));
	if (!hp.a)
		return 0;
	if (prefilter(vs, q, qdim, &hp) != 0) {
		free(hp.a);
		return 0;
	}

	rec_rank_t *rk = rec_rank_new(k, min_sim);
	if (!rk) {
		free(hp.a);
		return 0;
	}
	for (size_t i = 0; i < hp.n; i++) {
		sepal_blob_hdr_t h;
		const uint8_t *b = lookup_blob(vs, hp.a[i].r, &h);
		if (!b || h.dim > qdim)
			continue;
		float sc = sepal_cosine(q, blob_floats(b), h.dim);
		rec_rank_push(rk, hp.a[i].r, sc);
	}
	free(hp.a);

	size_t cap = k;
	rec_ref_t *refs = malloc(cap * sizeof(*refs));
	float *scs = malloc(cap * sizeof(*scs));
	size_t n = 0;
	if (refs && scs)
		n = rec_rank_sorted(rk, refs, scs);
	for (size_t i = 0; i < n; i++) {
		out[i].ref = refs[i];
		out[i].score = scs[i];
	}
	free(scs);
	free(refs);
	rec_rank_free(rk);
	return n;
}

/* ----------------------------------------------------------------------.
 * Kernel adapters                                                        */

int
sepal_fill_approx(sepal_vecstore_t *vs, const float *q, size_t qdim,
                  size_t m, float min_sim, rec_set_t *out)
{
	(void)min_sim;
	if (!vs || !q || !out)
		return -1;
	if (qdim < 1 || qdim > SEPAL_VEC_MAX)
		return -1;
	if (m == 0)
		m = vs->n;
	if (m > vs->n)
		m = vs->n;

	cand_heap_t hp = { 0 };
	if (m > 0) {
		hp.m = m;
		hp.a = malloc(m * sizeof(cand_t));
		if (!hp.a)
			return -1;
		if (prefilter(vs, q, qdim, &hp) != 0) {
			free(hp.a);
			return -1;
		}
		for (size_t i = 0; i < hp.n; i++)
			rec_set_push(out, hp.a[i].r);
		free(hp.a);
	}
	rec_set_seal(out);
	float bound = (m >= vs->n) ? 1.0f : (float)m / (float)vs->n;
	rec_set_set_approx(out, REC_SET_APPROX, bound);
	return 0;
}

int
sepal_rank(struct sepal_rank_ctx *ctx, rec_ref_t ref, float *score)
{
	if (!ctx || !ctx->vs || !ctx->q || !score)
		return -1;
	if (ctx->qdim < 1 || ctx->qdim > SEPAL_VEC_MAX)
		return -1;
	sepal_blob_hdr_t h;
	const uint8_t *b = lookup_blob(ctx->vs, ref, &h);
	if (!b || h.dim > ctx->qdim)
		return -1;
	float s = sepal_cosine(ctx->q, blob_floats(b), h.dim);
	if (s < ctx->min_sim)
		return -1;
	*score = s;
	return 0;
}