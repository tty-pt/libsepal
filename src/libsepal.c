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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xxhash.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ----------------------------------------------------------------------.
 * VEC1 blob (also see sepal.h): 16 + 8W + 4*dim bytes, little-endian.      */

#define BLOB_MAGIC0 'V'
#define BLOB_MAGIC1 'E'
#define BLOB_MAGIC2 'C'
#define BLOB_MAGIC3 '1'
#define BLOB_VERSION 1
#define BLOB_HDR_LEN 16

static void sepal_sketch_words(const float *v, size_t n, uint64_t *words);

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
	sepal_sketch_words(v, full_dim, sk);

	float *fs = (float *)(void *)(buf + BLOB_HDR_LEN + 8 * w);
	for (size_t i = 0; i < dim; i++)
		fs[i] = v[i];
	return len;
}

/* ----------------------------------------------------------------------.
 * Store                                                                */

struct sepal_vecstore {
	uint32_t hd;
	size_t   n;       /* entry count (restored on open, maintained) */
	size_t   idx_n;   /* rows in the flat search index (== n while enabled) */
	size_t   cap;     /* flat-index row capacity */
	size_t   store_w; /* u64 words per flat row */
	rec_ref_t *idx_ref;
	uint64_t *idx_sk;
	uint8_t  *idx_w;  /* per-row stored word count */
	int      idx_valid; /* 1 = index mirrors the map; 0 = disabled (fallback) */
	uint64_t *hkey;   /* open-addressing slot map: ref -> row */
	uint32_t *hval;
	uint8_t  *hused;
	size_t   hmcap;
};

static uint32_t sepal_kt = QM_MISS;
static uint32_t sepal_vt = QM_MISS;

/* ---------- flat index: ref->row slot map (XXH32 keyed like qmap) ------ */

#define SLOT_SEED 0xD15EA5E1U

static uint32_t
ref_hash(uint64_t x)
{
	return XXH32(&x, sizeof(x), SLOT_SEED);
}

static void
slot_free(sepal_vecstore_t *vs)
{
	free(vs->hkey);
	free(vs->hval);
	free(vs->hused);
	vs->hkey = NULL;
	vs->hval = NULL;
	vs->hused = NULL;
	vs->hmcap = 0;
}

static int
slot_grow(sepal_vecstore_t *vs)
{
	size_t ncap = vs->hmcap ? vs->hmcap * 2 : 16;
	uint64_t *nk = calloc(ncap, sizeof(*nk));
	uint32_t *nv = malloc(ncap * sizeof(*nv));
	uint8_t  *nu = calloc(ncap, sizeof(*nu));
	if (!nk || !nv || !nu) {
		free(nk); free(nv); free(nu);
		return -1;
	}
	size_t ocap = vs->hmcap;
	for (size_t i = 0; i < ocap; i++) {
		if (!vs->hused[i])
			continue;
		size_t j = ref_hash(vs->hkey[i]) & (ncap - 1);
		while (nu[j])
			j = (j + 1) & (ncap - 1);
		nk[j] = vs->hkey[i];
		nv[j] = vs->hval[i];
		nu[j] = 1;
	}
	slot_free(vs);
	vs->hkey = nk; vs->hval = nv; vs->hused = nu; vs->hmcap = ncap;
	return 0;
}

/* Insert or update (ref -> row). Returns 0 ok, -1 on allocation failure. */
static int
slot_put(sepal_vecstore_t *vs, rec_ref_t ref, uint32_t row)
{
	if (!vs->hused) {
		vs->hmcap = 0;
		if (slot_grow(vs) != 0)
			return -1;
	}
	size_t i = ref_hash(ref) & (vs->hmcap - 1);
	for (;;) {
		if (!vs->hused[i]) {
			vs->hkey[i] = ref;
			vs->hval[i] = row;
			vs->hused[i] = 1;
			return 0;
		}
		if (vs->hkey[i] == ref) {
			vs->hval[i] = row;
			return 0;
		}
		i = (i + 1) & (vs->hmcap - 1);
		if (i == (ref_hash(ref) & (vs->hmcap - 1)))
			break; /* table full */
	}
	if (slot_grow(vs) != 0)
		return -1;
	return slot_put(vs, ref, row);
}

static int
slot_find(const sepal_vecstore_t *vs, rec_ref_t ref, uint32_t *row)
{
	if (!vs->hused)
		return 0;
	size_t i = ref_hash(ref) & (vs->hmcap - 1);
	for (;;) {
		if (!vs->hused[i])
			return 0;
		if (vs->hkey[i] == ref) {
			if (row)
				*row = vs->hval[i];
			return 1;
		}
		i = (i + 1) & (vs->hmcap - 1);
		if (i == (ref_hash(ref) & (vs->hmcap - 1)))
			return 0;
	}
}

/* Rebuild ref->row from the flat rows (after a swap-remove deletion). */
static int
slot_rebuild(sepal_vecstore_t *vs)
{
	slot_free(vs);
	for (size_t i = 0; i < vs->idx_n; i++)
		if (slot_put(vs, vs->idx_ref[i], (uint32_t)i) != 0)
			return -1;
	return 0;
}

/* ---------- flat index: row arrays ---------- */

static void
idx_free(sepal_vecstore_t *vs)
{
	free(vs->idx_ref);  vs->idx_ref = NULL;
	free(vs->idx_sk);   vs->idx_sk = NULL;
	free(vs->idx_w);    vs->idx_w = NULL;
	vs->idx_n = 0;
	vs->cap = 0;
	vs->store_w = 0;
	slot_free(vs);
}

static int
idx_ensure(sepal_vecstore_t *vs, size_t need)
{
	if (need <= vs->cap)
		return 0;
	size_t ncap = vs->cap ? vs->cap : 64;
	while (ncap < need)
		ncap += ncap;
	rec_ref_t *nr = realloc(vs->idx_ref, ncap * sizeof(*nr));
	uint64_t *ns = NULL;
	if (vs->store_w > 0)
		ns = realloc(vs->idx_sk, ncap * vs->store_w * sizeof(*ns));
	uint8_t *nw = realloc(vs->idx_w, ncap);
	if (!nr || (vs->store_w > 0 && !ns) || !nw) {
		free(nr); free(ns); free(nw);
		return -1;
	}
	vs->idx_ref = nr;
	vs->idx_sk = ns;
	vs->idx_w = nw;
	vs->cap = ncap;
	return 0;
}

/* Grow the row stride (rare: a put with a larger dim); rewrites rows. */
static int
idx_resize_words(sepal_vecstore_t *vs, size_t new_w)
{
	if (new_w <= vs->store_w)
		return 0;
	if (vs->store_w > 0 && vs->idx_n > 0) {
		uint64_t *nsk = calloc(vs->cap, new_w * sizeof(*nsk));
		if (!nsk)
			return -1;
		for (size_t i = 0; i < vs->idx_n; i++)
			for (size_t j = 0; j < vs->idx_w[i] && j < vs->store_w; j++)
				nsk[i * new_w + j] = vs->idx_sk[i * vs->store_w + j];
		free(vs->idx_sk);
		vs->idx_sk = nsk;
	} else {
		vs->idx_sk = calloc(vs->cap, new_w * sizeof(*vs->idx_sk));
		if (!vs->idx_sk)
			return -1;
	}
	vs->store_w = new_w;
	return 0;
}

/* Append a parsed blob's row. Blob must have passed blob_parse already. */
static int
idx_embed(sepal_vecstore_t *vs, rec_ref_t ref, const uint8_t *blob,
          const sepal_blob_hdr_t *h)
{
	if (idx_ensure(vs, vs->idx_n + 1) != 0)
		return -1;
	if (idx_resize_words(vs, h->sketch_words) != 0)
		return -1;
	size_t row = vs->idx_n;
	const uint64_t *sk = (const uint64_t *)(const void *)(blob + BLOB_HDR_LEN);
	for (size_t j = 0; j < h->sketch_words; j++)
		vs->idx_sk[row * vs->store_w + j] = sk[j];
	vs->idx_ref[row] = ref;
	vs->idx_w[row] = (uint8_t)h->sketch_words;
	vs->idx_n++;
	return slot_put(vs, ref, (uint32_t)row);
}

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
vs->idx_valid = 1;

/* an existing file already holds entries: restore the count and mirror
 * every valid blob into the flat search index (atomicity: any unparsable
 * row disables the index → searches fall back to the map walk) */
{
	uint32_t cur = qmap_iter(hd, NULL, 0);
	const void *k, *v;
	while (qmap_next(&k, &v, cur)) {
		sepal_blob_hdr_t h;
		vs->n++;
		if (!vs->idx_valid)
			continue;
		if (blob_parse(v, (size_t)1 << 30, &h) != 0) {
			idx_free(vs);
			vs->idx_valid = 0;
			continue;
		}
		rec_ref_t ref;
		memcpy(&ref, k, sizeof(ref));
		if (idx_embed(vs, ref, v, &h) != 0) {
			idx_free(vs);
			vs->idx_valid = 0;
		}
	}
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
	idx_free(vs);
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
	if (qmap_count(vs->hd, &ref) == 0) {
		free(buf);
		return -1;
	}
	if (!was_present)
		vs->n++;

	/* Flat index (T2): replace updates the row in place; new refs append. */
	if (vs->idx_valid) {
		sepal_blob_hdr_t h;
		if (blob_parse(buf, len, &h) == 0) {
			uint32_t row;
			if (was_present && slot_find(vs, ref, &row)) {
				const uint64_t *sk = (const uint64_t *)(const void *)
					(buf + BLOB_HDR_LEN);
				if (idx_resize_words(vs, h.sketch_words) != 0) {
					free(buf);
					vs->idx_valid = 0;
					return 0;
				}
				for (size_t j = 0; j < h.sketch_words; j++)
					vs->idx_sk[row * vs->store_w + j] = sk[j];
				vs->idx_w[row] = (uint8_t)h.sketch_words;
			} else if (!was_present) {
				if (idx_embed(vs, ref, buf, &h) != 0) {
					vs->idx_valid = 0;
				}
			} else {
				/* index lost the ref (shouldn't happen) — disable */
				vs->idx_valid = 0;
			}
		} else {
			vs->idx_valid = 0;
		}
	}
	free(buf);
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
	if (vs->idx_valid) {
		uint32_t row;
		if (!slot_find(vs, ref, &row)) {
			vs->idx_valid = 0;          /* index lost sync — disable */
			return 0;
		}
		size_t last = vs->idx_n - 1;
		if (row != last) {              /* swap-remove the row */
			vs->idx_ref[row] = vs->idx_ref[last];
			vs->idx_w[row] = vs->idx_w[last];
			for (size_t j = 0; j < vs->store_w; j++)
				vs->idx_sk[row * vs->store_w + j] =
					vs->idx_sk[last * vs->store_w + j];
		}
		vs->idx_n--;
		/* slot map is rare-path rebuilt from the surviving rows */
		if (slot_rebuild(vs) != 0)
			vs->idx_valid = 0;
	}
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

int
sepal_index_validate(const sepal_vecstore_t *vs)
{
	if (!vs)
		return 0;
	if (!vs->idx_valid)
		return 0;                 /* index disabled — map scan is authoritative */
	if (vs->n != vs->idx_n)
		return 1;
	if (vs->idx_n > 0 &&
	    (!vs->idx_ref || !vs->idx_sk || !vs->idx_w || vs->cap < vs->idx_n ||
	     vs->store_w == 0))
		return 1;
	for (size_t i = 0; i < vs->idx_n; i++) {
		if (vs->idx_w[i] < 1 || vs->idx_w[i] > vs->store_w ||
		    vs->idx_w[i] > SEPAL_VEC_MAX / 64)
			return 1;
	}
	if (vs->hused) {
		size_t used = 0;
		for (size_t i = 0; i < vs->hmcap; i++) {
			if (!vs->hused[i])
				continue;
			used++;
			if (vs->hval[i] >= vs->idx_n ||
			    vs->idx_ref[vs->hval[i]] != vs->hkey[i])
				return 1;
		}
		if (used != vs->idx_n)
			return 1;
	}
	return 0;
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

#ifdef __AVX2__
/* AVX2 popcount via 2x nibble lookup table (reads 4 u64 in bounds). */
static uint64_t
sepal_popcnt_u64x4(const uint64_t *q, const uint64_t *sk)
{
	__m256i lookup = _mm256_setr_epi8(
		0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
		0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
	__m256i low = _mm256_set1_epi8(0x0f);
	__m256i x = _mm256_xor_si256(
		_mm256_loadu_si256((const __m256i *)q),
		_mm256_loadu_si256((const __m256i *)sk));
	__m256i lo = _mm256_and_si256(x, low);
	__m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), low);
	__m256i cnt = _mm256_add_epi8(
		_mm256_shuffle_epi8(lookup, lo),
		_mm256_shuffle_epi8(lookup, hi));
	__m128i lo16 = _mm256_castsi256_si128(cnt);
	__m128i hi16 = _mm256_extracti128_si256(cnt, 1);
	__m128i sum16 = _mm_add_epi8(lo16, hi16);
	__m128i sum32 = _mm_sad_epu8(sum16, _mm_setzero_si128());
	return (uint64_t)(_mm_cvtsi128_si64(sum32) + _mm_extract_epi64(sum32, 1));
}

/* Hamming distance over n words: 4-word AVX2 groups + scalar tail. Only
 * touches [0, n) — safe on the flat index's last (partial) row. */
static uint64_t
sepal_popcnt_range(const uint64_t *q, const uint64_t *sk, size_t n)
{
	uint64_t sum = 0;
	size_t i = 0;
	for (; i + 4 <= n; i += 4)
		sum += sepal_popcnt_u64x4(q + i, sk + i);
	for (; i < n; i++)
		sum += (uint64_t)__builtin_popcountll(q[i] ^ sk[i]);
	return sum;
}

/* Sign-bit sketch words over v[0..n): AVX2 movemask, 64 floats per full
 * word; the tail word (< 64 dims) is scalar so reads never pass n. */
static void
sepal_sketch_words(const float *v, size_t n, uint64_t *words)
{
	size_t full = n / 64;
	for (size_t wi = 0; wi < full; wi++) {
		const float *p = v + wi * 64;
		__m256i m0  = _mm256_castps_si256(_mm256_loadu_ps(p));
		__m256i m4  = _mm256_castps_si256(_mm256_loadu_ps(p + 8));
		__m256i m8  = _mm256_castps_si256(_mm256_loadu_ps(p + 16));
		__m256i m12 = _mm256_castps_si256(_mm256_loadu_ps(p + 24));
		__m256i m16 = _mm256_castps_si256(_mm256_loadu_ps(p + 32));
		__m256i m20 = _mm256_castps_si256(_mm256_loadu_ps(p + 40));
		__m256i m24 = _mm256_castps_si256(_mm256_loadu_ps(p + 48));
		__m256i m28 = _mm256_castps_si256(_mm256_loadu_ps(p + 56));
		words[wi] =
			(uint64_t)(uint32_t)_mm256_movemask_ps((__m256)m0) |
			(uint64_t)(uint32_t)_mm256_movemask_ps((__m256)m4) << 8 |
			(uint64_t)(uint32_t)_mm256_movemask_ps((__m256)m8) << 16 |
			(uint64_t)(uint32_t)_mm256_movemask_ps((__m256)m12) << 24 |
			(uint64_t)(uint32_t)_mm256_movemask_ps((__m256)m16) << 32 |
			(uint64_t)(uint32_t)_mm256_movemask_ps((__m256)m20) << 40 |
			(uint64_t)(uint32_t)_mm256_movemask_ps((__m256)m24) << 48 |
			(uint64_t)(uint32_t)_mm256_movemask_ps((__m256)m28) << 56;
	}
	size_t rem = n % 64;
	if (rem) {
		const float *p = v + full * 64;
		uint64_t word = 0;
		for (size_t i = 0; i < rem; i++)
			if (__builtin_signbit(p[i]))
				word |= 1ULL << i;
		words[full] = word;
	}
}
#else /* !__AVX2__ */
static uint64_t
sepal_popcnt_range(const uint64_t *q, const uint64_t *sk, size_t n)
{
	uint64_t sum = 0;
	for (size_t i = 0; i < n; i++)
		sum += (uint64_t)__builtin_popcountll(q[i] ^ sk[i]);
	return sum;
}

static void
sepal_sketch_words(const float *v, size_t n, uint64_t *words)
{
	size_t need = (n + 63) / 64;
	for (size_t wi = 0; wi < need; wi++) {
		uint64_t word = 0;
		size_t base = wi * 64;
		size_t hi = base + 64 < n ? base + 64 : n;
		for (size_t i = base; i < hi; i++)
			if (__builtin_signbit(v[i]))
				word |= 1ULL << (i - base);
		words[wi] = word;
	}
}
#endif /* __AVX2__ */

int
sepal_sketch(const float *v, size_t n, uint64_t *words, size_t nwords)
{
	if (!v || !words || n < 1 || n > SEPAL_VEC_MAX)
		return -1;
	size_t need = (n + 63) / 64;
	if (nwords < need)
		return -1;
	sepal_sketch_words(v, n, words);
	return 0;
}

/* ----------------------------------------------------------------------.
 * Two-stage search                                                       */

/* Frozen-norm rerank score: cos(q, bf[0..d)) from a single fused dot pass
 * using norms frozen at put (the blob header) and a query prefix norm —
 * semantically equal to sepal_cosine within float rounding (D1), while
 * cutting stage-2 to one pass over the stored floats. */
static float
rerank_score(const float *q, const float *bf, size_t d, float qnorm_d,
             float stored_norm)
{
	double dot = 0.0;
	for (size_t i = 0; i < d; i++)
		dot += (double)q[i] * (double)bf[i];
	if (qnorm_d <= 0.0f || stored_norm <= 0.0f)
		return 0.0f;
	return (float)(dot / ((double)qnorm_d * (double)stored_norm));
}

/* Single-candidate variant used by sepal_rank: dot and the query prefix
 * norm share one pass over d (nothing extra vs sepal_cosine's fused pass —
 * nb is the frozen blob norm). */
static float
rerank_score_fused(const float *q, const float *bf, size_t d, float stored_norm)
{
	double dot = 0.0, qacc = 0.0;
	for (size_t i = 0; i < d; i++) {
		dot += (double)q[i] * (double)bf[i];
		qacc += (double)q[i] * (double)q[i];
	}
	float qn = (float)sqrt(qacc);
	if (qn <= 0.0f || stored_norm <= 0.0f)
		return 0.0f;
	return (float)(dot / ((double)qn * (double)stored_norm));
}

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

/* Scan the store, keeping the m smallest Hamming refs. Returns 0 ok,
 * -1 on allocation failure. T2: when the flat index is enabled the walk is
 * a tight linear scan of contiguous sketch rows (L2/L3-resident); otherwise
 * it falls back to the blob-map walk (kept for files whose rows fail parse
 * or when the index lost sync). Either way the candidate SET is identical. */
static int
prefilter(sepal_vecstore_t *vs, const float *q, size_t qdim,
          cand_heap_t *hp)
{
	uint64_t qw[SEPAL_VEC_MAX / 64];
	size_t wq = (qdim + 63) / 64;
	if (sepal_sketch(q, qdim, qw, wq) != 0)
		return -1;

	if (!vs->idx_valid) {
		uint32_t cur = qmap_iter(vs->hd, NULL, 0);
		const void *k, *v;
		while (qmap_next(&k, &v, cur)) {
			sepal_blob_hdr_t h;
			if (blob_parse(v, (size_t)1 << 30, &h) != 0)
				continue;
			const uint64_t *sk = (const uint64_t *)(const void *)
				((const uint8_t *)v + BLOB_HDR_LEN);
			size_t words = wq < h.sketch_words ? wq : h.sketch_words;
			uint64_t hd = sepal_popcnt_range(qw, sk, words);
			rec_ref_t ref;
			memcpy(&ref, k, sizeof(ref));
			heap_push(hp, (uint32_t)hd, ref);
		}
		qmap_fin(cur);
		return 0;
	}

	for (size_t row = 0; row < vs->idx_n; row++) {
		const uint64_t *sk = vs->idx_sk + (size_t)row * vs->store_w;
		size_t words = wq < vs->idx_w[row] ? wq : vs->idx_w[row];
		uint64_t hd = sepal_popcnt_range(qw, sk, words);
		heap_push(hp, (uint32_t)hd, vs->idx_ref[row]);
	}
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
	/* query prefix norms (double, archived to float) over the exact-stage
	 * dims — built once, indexed by each candidate's stored dim */
	size_t pq = qdim < SEPAL_EXACT_DIM ? qdim : SEPAL_EXACT_DIM;
	float qn[SEPAL_EXACT_DIM + 1];
	double qacc = 0.0;
	qn[0] = 0.0f;
	for (size_t i = 0; i < pq; i++) {
		qacc += (double)q[i] * (double)q[i];
		qn[i + 1] = (float)sqrt(qacc);
	}
	for (size_t i = 0; i < hp.n; i++) {
		sepal_blob_hdr_t h;
		const uint8_t *b = lookup_blob(vs, hp.a[i].r, &h);
		if (!b || h.dim > qdim)
			continue;
		float sc = rerank_score(q, blob_floats(b), h.dim, qn[h.dim],
		                        h.norm);
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
	float s = rerank_score_fused(ctx->q, blob_floats(b), h.dim, h.norm);
	if (s < ctx->min_sim)
		return -1;
	*score = s;
	return 0;
}
/* ---- rec_query axis registration (sepal / meaning) ---- */

struct rec_sepal_params {
	float  *q;       /* heap-owned query vector, freed never (see decode) */
	size_t  qdim;
	size_t  m;
	float   min_sim;
};

static int sepal_axis_fill(void *ctx, void *params, rec_set_t *out)
{
	sepal_vecstore_t *vs = ctx;
	const struct rec_sepal_params *p = params;

	if (!p || !p->q)
		return -1;
	return sepal_fill_approx(vs, p->q, p->qdim, p->m, p->min_sim, out);
}

static int sepal_axis_rank(void *ctx, void *params, rec_ref_t ref, float *score)
{
	const struct rec_sepal_params *p = params;
	struct sepal_rank_ctx sc;

	if (!p || !p->q)
		return -1;
	sc.vs = ctx;
	sc.q = p->q;
	sc.qdim = p->qdim;
	sc.min_sim = p->min_sim;
	return sepal_rank(&sc, ref, score);
}

/*
 * Decode "file=vecs.bin qdim=256 m=10 min_sim=0.5" into a heap-owned
 * rec_sepal_params (freed never — one-shot CLI process lifetime, matches
 * the other axis decode fns). `file` is a flat little-endian float32 blob
 * of exactly qdim floats (NOT a VEC1 blob — that format is for entries
 * inside a sepal_vecstore_t, not for a one-off CLI query vector). `file`
 * and `qdim` are required; `m` (candidate pool, 0 = library default) and
 * `min_sim` (default 0.0) are optional. NULL on missing/unreadable file,
 * missing qdim, or OOM.
 */
static void *sepal_axis_decode(const char *s)
{
	struct rec_sepal_params *p;
	char *buf, *cur;
	const char *file = NULL;
	size_t qdim = 0, m = 0;
	float min_sim = 0.0f;
	FILE *f;

	if (!s)
		return NULL;
	buf = malloc(strlen(s) + 1);
	if (!buf)
		return NULL;
	strcpy(buf, s);
	cur = buf;
	while (*cur) {
		char *key, *val;

		while (*cur == ' ')
			cur++;
		if (!*cur)
			break;
		key = cur;
		while (*cur && *cur != '=' && *cur != ' ')
			cur++;
		if (*cur != '=') {
			if (*cur)
				cur++;
			continue;
		}
		*cur++ = '\0';
		val = cur;
		while (*cur && *cur != ' ')
			cur++;
		if (*cur)
			*cur++ = '\0';
		if (!strcmp(key, "file"))
			file = val;
		else if (!strcmp(key, "qdim"))
			qdim = (size_t)atol(val);
		else if (!strcmp(key, "m"))
			m = (size_t)atol(val);
		else if (!strcmp(key, "min_sim"))
			min_sim = (float)atof(val);
	}
	if (!file || qdim == 0) {
		free(buf);
		return NULL;
	}
	p = calloc(1, sizeof(*p));
	if (!p) {
		free(buf);
		return NULL;
	}
	p->q = malloc(qdim * sizeof(float));
	if (!p->q) {
		free(p);
		free(buf);
		return NULL;
	}
	f = fopen(file, "rb");
	if (!f || fread(p->q, sizeof(float), qdim, f) != qdim) {
		if (f)
			fclose(f);
		free(p->q);
		free(p);
		free(buf);
		return NULL;
	}
	fclose(f);
	p->qdim = qdim;
	p->m = m;
	p->min_sim = min_sim;
	free(buf);
	return p;
}

__attribute__((constructor)) static void sepal_rec_axis_init(void)
{
	static const rec_axis_t sepal_axis = {
		"sepal", sepal_axis_fill, sepal_axis_rank, NULL, sepal_axis_decode
	};

	rec_axis_register(&sepal_axis);
}

/*
 * rec_axis_open convention (PLAN-REC-QUERY.md §4.3, optional CLI-open
 * convention, not part of libqmap's core rec_query registry API): spec
 * is the sepal_open() fname, or empty/NULL for a memory-only store.
 * Returns the sepal_vecstore_t* ctx directly (no cast needed, unlike
 * the joint/islet handle-widening axes) -- NULL on open failure.
 */
void *rec_axis_open(const char *spec)
{
	int err;

	return sepal_open(spec && *spec ? spec : NULL, &err);
}
