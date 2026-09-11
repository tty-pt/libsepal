/*
 * sepal.h — libsepal, the meaning (semantic) axis.
 *
 * A dim-tagged vector store with a two-stage ANN pipeline: Hamming prefilter
 * over a sign-bit sketch (top-m) followed by exact cosine rerank (top-k).
 * Built on the qmap persistent map and the recall kernel (rec.h). Follows the
 * same standard as libit / libgeo / libstoma (mk stack, ttypt headers).
 *
 * DESIGN CONTRACT
 *  - Refs are caller-opaque rec_ref_t (u64): the caller passes them at put
 *    time and owns ref<->schema mapping. The store never maps refs.
 *  - Per-blob dims: every vector stores its own exact-stage dim (the
 *    "matryoshka" prefix) = min(full_dim, SEPAL_EXACT_DIM). Vectors already
 *    <= SEPAL_EXACT_DIM dims are stored untruncated.
 *  - sepal_get NEVER truncates: asking for more dims than stored is fine
 *    (returns the stored count); asking for fewer returns 0 (mismatch).
 *  - Search is pure two-stage ANN. Exact brute force is the test-side
 *    reference and is NOT provided by the library (exactness is a kernel
 *    property, not a sepal feature).
 */
#ifndef TTYPT_SEPAL_H
#define TTYPT_SEPAL_H

#include <stddef.h>
#include <stdint.h>
#include <ttypt/rec.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Max full dimensionality accepted at put (any dim is truncated to this). */
#define SEPAL_VEC_MAX   2048

/** Matryoshka default: exact-stage dim cap (per-blob dim, not store config). */
#define SEPAL_EXACT_DIM 256

/** m knob default: 10x k when m==0 (capped at the store size). */
#define SEPAL_M_DEFAULT 10

/** Opaque vector store handle (qmap-backed; memory-only when opened NULL). */
typedef struct sepal_vecstore sepal_vecstore_t;

/** A search hit: (ref, cosine score). */
typedef struct sepal_hit {
	rec_ref_t ref;
	float     score;
} sepal_hit_t;

/* ---------------------------------------------------------------------.
 *  Store lifecycle
 *  - sepal_open(fname, &err): persistent sidecar store over <fname> (one
 *    map per file). fname == NULL -> memory-only store. err (may be NULL)
 *    receives 0 on success, -1 on open/create failure, -2 on memory or
 *    type-registration failure.
 *  - sepal_close persists via qmap_save() and never calls qmap_close():
 *    closing truncates the file to 0 (the same no-close invariant as mm).
 *    After sepal_close the store must not be used.
 * `--------------------------------------------------------------------- */
sepal_vecstore_t *sepal_open(const char *fname, int *err);
void              sepal_close(sepal_vecstore_t *vs);

/* ---------------------------------------------------------------------.
 *  Vectors (blobs)                                                     */
/** Store vector v (full_dim floats); dim stored = min(full_dim,
 *  SEPAL_EXACT_DIM). full_dim must be in [1, SEPAL_VEC_MAX]; -1 on error.
 *  Re-putting a ref replaces its blob. */
int sepal_put(sepal_vecstore_t *vs, rec_ref_t ref,
              const float *v, size_t full_dim);

/** Delete a ref; 0 if present, -1 if absent. */
int sepal_del(sepal_vecstore_t *vs, rec_ref_t ref);

/** Copy the stored exact-dim floats into out (space for >= dim). Returns
 *  the stored exact dim (>0). Returns 0 on missing ref, or when max < stored
 *  dim (mismatch — never truncates, per the design contract). */
size_t sepal_get(sepal_vecstore_t *vs, rec_ref_t ref,
                 float *out, size_t max);

/** Stored exact-stage dim of ref (0 if absent). */
size_t sepal_dim(sepal_vecstore_t *vs, rec_ref_t ref);

/** Full (sketch) dim of ref (0 if absent). */
size_t sepal_full_dim(sepal_vecstore_t *vs, rec_ref_t ref);

/** Number of vectors in the store. */
size_t sepal_n(const sepal_vecstore_t *vs);

/* ---------------------------------------------------------------------.
 *  Math helpers                                                        */

/** Column-normalized cosine similarity (mm_cosine semantics); 0 when either
 *  vector has zero norm. */
float sepal_cosine(const float *a, const float *b, size_t n);

/** Sign-bit sketch: nwords must be >= ceil(n/64) (else -1). words[w] bit b
 *  is the sign bit (IEEE bit 31) of v[w*64 + b], for dims w*64+b < n; the
 *  bits past n in the last word are zeroed. Returns 0 on success. */
int sepal_sketch(const float *v, size_t n, uint64_t *words, size_t nwords);

/* ---------------------------------------------------------------------.
 *  Two-stage ANN search
 *    stage 1: Hamming prefilter over sketches, keep best-m (m==0 ->
 *             10*k capped at N);
 *    stage 2: exact-cosine rerank of the m candidates, keep best-k above
 *             min_sim, ordered best-first, ties by ascending ref.
 *  qdim > SEPAL_VEC_MAX is rejected (returns 0). Returns hits written
 *  (<= k), best-first; never more than the candidate pool (m) or k. */
size_t sepal_search(sepal_vecstore_t *vs, const float *q, size_t qdim,
                    size_t k, float min_sim, size_t m, sepal_hit_t *out);

/* ---------------------------------------------------------------------.
 *  Recall-kernel adapters                                              */

/** Approximate set fill (kernel form): streams the top-m Hamming refs into
 *  out, seals it, then declares it approximate via rec_set_set_approx()
 *  with owed bound = (m >= n ? 1.0 : (float)m / n) — the oversampling
 *  heuristic (caller may override after the call; the setter exists).
 *  Additive: on entry out may already hold refs (result is their union,
 *  sealed). 0 on success; -1 on NULL args / bad dims. */
int sepal_fill_approx(sepal_vecstore_t *vs, const float *q, size_t qdim,
                      size_t m, float min_sim, rec_set_t *out);

/** Exact rerank context for sepal_rank (a rec_score_fn-style callback). */
struct sepal_rank_ctx {
	sepal_vecstore_t *vs;
	const float      *q;
	size_t            qdim;
	float             min_sim;
};

/** Exact cosine rerank of one ref. 0 on success (score written, >= min_sim);
 *  nonzero when ref is missing, dims mismatch, or score < min_sim (score is
 *  left untouched on failure). */
int sepal_rank(struct sepal_rank_ctx *ctx, rec_ref_t ref, float *score);

/* ---------------------------------------------------------------------.
 *  VEC1 blob (binary-level access; rarely used directly)
 *
 *  The serialized blob is the on-disk / in-map format of a stored vector:
 *      off  size  field
 *      0    4     magic 'V','E','C','1'  (legacy text blobs start with a digit)
 *      4    2     version (1)
 *      6    2     dim           (exact-stage dim = min(full_dim, SEPAL_EXACT_DIM))
 *      8    2     full_dim      (sketch source dim)
 *      10   2     sketch_words  (W = ceil(full_dim/64))
 *      12   4     norm          (float32, ||v[0..dim)||, frozen at put)
 *      16   8W    sketch        (W u64 sign words over the FULL-dim vector)
 *      16+8W 4*dim vec          (float32 truncated prefix)
 *  Total 16 + 8W + 4*dim  ->  1136 B at full 768 / exact 256.
 *  Little-endian. Garbage/truncated/wrong-magic blobs always decode to 0.
 */

typedef struct sepal_blob_hdr {
	unsigned dim;          /* exact-stage floats stored */
	unsigned full_dim;     /* sketch source dim */
	unsigned sketch_words; /* u64 words in the sketch */
	float    norm;         /* norm of v[0..dim) */
} sepal_blob_hdr_t;

/** Byte length of a valid VEC1 blob; 0 when invalid (bad magic/version,
 *  inconsistent fields) or truncated (declared length > avail). */
size_t sepal_blob_len(const uint8_t *blob, size_t avail);

/** Parse the VEC1 header; 0 ok, -1 invalid/truncated. */
int sepal_blob_hdr(const uint8_t *blob, size_t avail, sepal_blob_hdr_t *hdr);

/** Encode a vector as a VEC1 blob (sketch over full_dim, norm of the
 *  truncated prefix dim = min(full_dim, SEPAL_EXACT_DIM)). Writes at most
 *  avail bytes; returns the byte length written, or 0 on error (bad dims or
 *  avail too small). dim must be <= SEPAL_EXACT_DIM and <= full_dim. */
size_t sepal_blob_put(uint8_t *buf, size_t avail,
                      const float *v, size_t full_dim, size_t dim);

#ifdef __cplusplus
}
#endif

#endif /* TTYPT_SEPAL_H */