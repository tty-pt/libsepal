/* test_blob_props.c — random byte blobs must never be misread.
 *
 * Property: for any byte buffer, sepal_blob_len() returns either 0 (refused)
 * or a consistent length (<= avail, identical on re-parse); sepal_blob_hdr()
 * agrees. Valid round-trips from sepal_blob_put() always survive slicing. */

#include "../test_common.h"

static void
randomize(uint8_t *b, size_t n, sepal_rng_t *r)
{
	for (size_t i = 0; i < n; i++)
		b[i] = (uint8_t)(rng_next(r) & 0xFF);
}

static void
test_blob_random_never_misreads(void)
{
	printf("=== blob props: random buffers never misread ===\n");
	sepal_rng_t r = { 71 };
	uint8_t buf[1600];

	for (size_t iter = 0; iter < 20000; iter++) {
		size_t len = (size_t)(rng_next(&r) % sizeof(buf));
		randomize(buf, len, &r);

		/* force the magic so we exercise the field logic, not the fast refuse */
		if ((iter & 3) == 0 && len >= 16) {
			buf[0] = 'V'; buf[1] = 'E'; buf[2] = 'C'; buf[3] = '1';
		}
		size_t l = sepal_blob_len(buf, len);
		if (l != 0) {
			ASSERT(l <= len, "parsed length never exceeds avail");
			ASSERT(l >= 16, "parsed length never below header");
			ASSERT_EQ(sepal_blob_len(buf, len), l);       /* deterministic */
			sepal_blob_hdr_t h;
			ASSERT_EQ(sepal_blob_hdr(buf, len, &h), 0);
			/* consistency of the parity: re-parse from the exact length */
			ASSERT_EQ(sepal_blob_len(buf, l), l);
			/* truncation must refuse */
			if (l > 16)
				ASSERT_EQ(sepal_blob_len(buf, l - 1), 0);
		} else {
			sepal_blob_hdr_t h;
			ASSERT_NE(sepal_blob_hdr(buf, len, &h), 0);
		}
	}
}

static void
test_blob_valid_roundtrip_survives_slicing(void)
{
	printf("=== blob props: valid blobs survive slicing ===\n");
	sepal_rng_t r = { 72 };
	uint8_t buf[1600];
	float v[SEPAL_EXACT_DIM];

	for (size_t dim = 1; dim <= SEPAL_EXACT_DIM; dim += 17) {
		size_t full = dim + 512;
		rng_unit_vector(&r, v, SEPAL_EXACT_DIM);
		size_t l = sepal_blob_put(buf, sizeof(buf), v, full, dim);
		ASSERT_NE(l, 0);
		ASSERT_EQ(sepal_blob_len(buf, sizeof(buf)), l);
		/* exact-length slice still parses; anything shorter refuses */
		ASSERT_EQ(sepal_blob_len(buf, l), l);
		ASSERT(l > 16 && sepal_blob_len(buf, l - 1) == 0, "truncated refuses");
	}
}

int
main(void)
{
	test_blob_random_never_misreads();
	test_blob_valid_roundtrip_survives_slicing();
	return test_summary();
}