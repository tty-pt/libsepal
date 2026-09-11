/* test_sketch.c — sign-bit sketches: known words, hamming, bounds. */

#include "../test_common.h"

static inline unsigned
popcount64(uint64_t x)
{
	return (unsigned)__builtin_popcountll(x);
}

static uint64_t
hamming(const uint64_t *a, const uint64_t *b, size_t n)
{
	uint64_t d = 0;
	for (size_t i = 0; i < n; i++)
		d += popcount64(a[i] ^ b[i]);
	return d;
}

static void
test_sketch_known(void)
{
	printf("=== sketch: known vectors → known words ===\n");
	{
		/* [-,+,-,+] → bits 0 and 2 set → 0b0101 = 5 */
		const float v[] = { -1.0f, 1.0f, -1.0f, 1.0f };
		uint64_t w[1];
		ASSERT_EQ(sepal_sketch(v, 4, w, 1), 0);
		ASSERT_EQ(w[0], 0b0101ULL);
	}
	{
		/* all positive → no sign bits */
		const float v[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
		uint64_t w[1];
		ASSERT_EQ(sepal_sketch(v, 6, w, 1), 0);
		ASSERT_EQ(w[0], 0ULL);
	}
	{
		/* all negative → every covered bit set, padding zeroed */
		float v[70];
		memset(v, 0, sizeof(v));
		for (size_t i = 0; i < 70; i++)
			v[i] = -1.0f;
		uint64_t w[2];
		ASSERT_EQ(sepal_sketch(v, 70, w, 2), 0);
		ASSERT_EQ(w[0], ~0ULL);
		ASSERT_EQ(w[1], 0x3fULL);           /* bits 0..5 set, 6..63 zero */
	}
	{
		/* sign of -0.0 is a set bit */
		const float v[] = { -0.0f, 0.0f };
		uint64_t w[1];
		ASSERT_EQ(sepal_sketch(v, 2, w, 1), 0);
		ASSERT_EQ(w[0], 1ULL);
	}
}

static void
test_sketch_bounds(void)
{
	printf("=== sketch: nwords too small → -1 ===\n");
	float v[768];
	memset(v, 0, sizeof(v));
	uint64_t w[12];
	ASSERT_EQ(sepal_sketch(v, 768, w, 12), 0);
	ASSERT_EQ(sepal_sketch(v, 768, w, 11), -1);
	ASSERT_EQ(sepal_sketch(v, 768, w, 0), -1);
	ASSERT_EQ(sepal_sketch(NULL, 768, w, 12), -1);
	ASSERT_EQ(sepal_sketch(v, 0, w, 12), -1);
	ASSERT_EQ(sepal_sketch(v, 768, NULL, 12), -1);
}

static void
test_sketch_hamming_symmetric(void)
{
	printf("=== sketch: hamming of known pairs ===\n");
	const float a[] = { -1, 1, -1, 1, -1, 1, -1, 1 };
	const float b[] = { 1, -1, 1, -1, 1, -1, 1, -1 };
	uint64_t wa[1], wb[1], wp[1];
	ASSERT_EQ(sepal_sketch(a, 8, wa, 1), 0);
	ASSERT_EQ(sepal_sketch(b, 8, wb, 1), 0);
	ASSERT_EQ(sepal_sketch(a, 8, wp, 1), 0);
	ASSERT_EQ(hamming(wa, wb, 1), 8);
	ASSERT_EQ(hamming(wa, wp, 1), 0);

	const float c[] = { -1, -1, 1, 1, -1, -1, 1, 1 };
	uint64_t wc[1];
	ASSERT_EQ(sepal_sketch(c, 8, wc, 1), 0);
	ASSERT_EQ(hamming(wa, wc, 1), 4);
}

int
main(void)
{
	test_sketch_known();
	test_sketch_bounds();
	test_sketch_hamming_symmetric();
	return test_summary();
}