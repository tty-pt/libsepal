/* test_blob.c — VEC1 blob format: encode/decode, validation, byte length. */

#include "../test_common.h"

static void
test_roundtrip_fields(void)
{
	printf("=== blob: round-trip fields (768/256 → 1136 bytes) ===\n");

	float v[768];
	sepal_rng_t r = { 42 };
	rng_unit_vector(&r, v, 768);

	uint8_t buf[2048];
	size_t len = sepal_blob_put(buf, sizeof(buf), v, 768, 256);
	ASSERT_EQ(len, 1136);                 /* 16 + 8*12 + 4*256 */

	sepal_blob_hdr_t hdr;
	ASSERT_EQ(sepal_blob_hdr(buf, sizeof(buf), &hdr), 0);
	ASSERT_EQ(hdr.dim, 256);
	ASSERT_EQ(hdr.full_dim, 768);
	ASSERT_EQ(hdr.sketch_words, 12);

	double s = 0.0;
	for (int i = 0; i < 256; i++)
		s += (double)v[i] * (double)v[i];
	ASSERT_NEAR(hdr.norm, sqrt(s), 1e-5);

	/* floats at offset 16 + 8*12 == bit-exact vs input prefix */
	const float *pf = (const float *)(buf + 16 + 8 * 12);
	ASSERT(memcmp(pf, v, 256 * sizeof(float)) == 0, "prefix floats bit-exact");
}

static void
test_blob_put_validation(void)
{
	printf("=== blob: encoding validation ===\n");

	float v[16] = { 0 };
	uint8_t buf[512];
	memset(buf, 0, sizeof(buf));          /* blob_len reads raw bytes */
	ASSERT_EQ(sepal_blob_put(NULL, sizeof(buf), v, 16, 16), 0);
	ASSERT_EQ(sepal_blob_put(buf, sizeof(buf), NULL, 16, 16), 0);
	ASSERT_EQ(sepal_blob_put(buf, sizeof(buf), v, 0, 16), 0);       /* full 0 */
	ASSERT_EQ(sepal_blob_put(buf, sizeof(buf), v, 16, 0), 0);       /* dim 0 */
	ASSERT_EQ(sepal_blob_put(buf, sizeof(buf), v, 16, 17), 0);      /* dim > full */
	ASSERT_EQ(sepal_blob_put(buf, sizeof(buf), v, 16, 257), 0);     /* dim > 256 */
	ASSERT_EQ(sepal_blob_put(buf, sizeof(buf), v, 2049, 256), 0);   /* full > max */
	ASSERT_EQ(sepal_blob_put(buf, 100, v, 768, 256), 0);            /* avail small */

	ASSERT_EQ(sepal_blob_len(buf, sizeof(buf)), 0);                 /* buf untouched? */
}

static void
test_decode_rejects_garbage(void)
{
	printf("=== blob: garbage/truncated/wrong magic/version → 0 ===\n");

	float v[16];
	sepal_rng_t r = { 7 };
	rng_unit_vector(&r, v, 16);
	uint8_t buf[512];
	size_t len = sepal_blob_put(buf, sizeof(buf), v, 16, 16);
	ASSERT_NE(len, 0);
	ASSERT_EQ(sepal_blob_len(buf, len), len);          /* exact round trip */
	ASSERT_EQ(sepal_blob_len(buf, len - 1), 0);        /* truncated */

	uint8_t junk[128];
	memset(junk, 0xAB, sizeof(junk));
	ASSERT_EQ(sepal_blob_len(junk, sizeof(junk)), 0);
	sepal_blob_hdr_t hdr;
	ASSERT_NE(sepal_blob_hdr(junk, sizeof(junk), &hdr), 0);

	/* wrong magic */
	uint8_t wrong[512];
	memcpy(wrong, buf, len);
	wrong[0] = 'A';
	ASSERT_EQ(sepal_blob_len(wrong, len), 0);

	/* wrong version */
	memcpy(wrong, buf, len);
	wrong[4] = 2;
	ASSERT_EQ(sepal_blob_len(wrong, len), 0);

	/* legacy-text blob: starts with a digit */
	const uint8_t text[] = { '0', '.', '5', ' ', '0', '.', '2', '5', '\n' };
	ASSERT_EQ(sepal_blob_len(text, sizeof(text)), 0);
}

static void
test_decode_rejects_inconsistent_fields(void)
{
	printf("=== blob: inconsistent fields → 0 ===\n");

	float v[16];
	sepal_rng_t r = { 3 };
	rng_unit_vector(&r, v, 16);
	uint8_t buf[512];
	size_t len = sepal_blob_put(buf, sizeof(buf), v, 16, 16);
	ASSERT_NE(len, 0);

	/* dim > full_dim */
	memcpy(&buf[8], &(uint16_t){ 8 }, 2);
	ASSERT_EQ(sepal_blob_len(buf, sizeof(buf)), 0);

	/* dim == 0 */
	memcpy(&buf[6], &(uint16_t){ 0 }, 2);
	ASSERT_EQ(sepal_blob_len(buf, sizeof(buf)), 0);
}

static void
test_sketch_words_field(void)
{
	printf("=== blob: sketch_words must equal ceil(full/64) ===\n");

	float v[128];
	sepal_rng_t r = { 5 };
	rng_unit_vector(&r, v, 128);
	uint8_t buf[1024];
	size_t len = sepal_blob_put(buf, sizeof(buf), v, 128, 128);
	ASSERT_NE(len, 0);

	memcpy(&buf[10], &(uint16_t){ 1 }, 2);   /* 1 != ceil(128/64)=2 */
	ASSERT_EQ(sepal_blob_len(buf, sizeof(buf)), 0);
}

int
main(void)
{
	test_roundtrip_fields();
	test_blob_put_validation();
	test_decode_rejects_garbage();
	test_decode_rejects_inconsistent_fields();
	test_sketch_words_field();
	return test_summary();
}