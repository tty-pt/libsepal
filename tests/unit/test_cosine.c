/* test_cosine.c — sepal_cosine correctness vs a naive double reference. */

#include "../test_common.h"

static void
test_cosine_basics(void)
{
	printf("=== cosine: orthogonal / parallel / opposite / zero-norm ===\n");
	{
		const float a[] = { 1, 0, 0 }, b[] = { 0, 1, 0 };
		ASSERT_NEAR(sepal_cosine(a, b, 3), 0.0f, 1e-6);
	}
	{
		const float a[] = { 1, 2, 3 }, b[] = { 2, 4, 6 };
		ASSERT_NEAR(sepal_cosine(a, b, 3), 1.0f, 1e-5);
	}
	{
		const float a[] = { 1, 1, 0 }, b[] = { -1, -1, 0 };
		ASSERT_NEAR(sepal_cosine(a, b, 3), -1.0f, 1e-5);
	}
	{
		const float a[] = { 0, 0, 0 }, b[] = { 1, 0, 0 };
		ASSERT_NEAR(sepal_cosine(a, b, 3), 0.0f, 0.0f);  /* zero norm → 0 */
	}
	{
		const float a[] = { 1, 0, 0 }, b[] = { 0, 0, 0 };
		ASSERT_NEAR(sepal_cosine(a, b, 3), 0.0f, 0.0f);
	}
	{
		const float a[] = { 1.5f, -0.5f, 0.25f }, b[] = { 0.0f, 1.0f, -0.75f };
		ASSERT_NEAR(sepal_cosine(a, b, 3), ref_cosine(a, b, 3), 1e-5);
	}
}

static void
test_cosine_random_parity(void)
{
	printf("=== cosine: random parity vs reference ===\n");
	sepal_rng_t r = { 42 };
	for (size_t n = 1; n <= 64; n++) {
		float a[64], b[64];
		double sa = 0, sb = 0;
		for (size_t i = 0; i < n; i++) {
			a[i] = rng_unit(&r);
			b[i] = rng_unit(&r);
			sa += (double)a[i] * (double)a[i];
			sb += (double)b[i] * (double)b[i];
		}
		if (sa <= 0 || sb <= 0)
			continue;
		ASSERT_NEAR(sepal_cosine(a, b, n), ref_cosine(a, b, n), 1e-4);
	}
}

int
main(void)
{
	test_cosine_basics();
	test_cosine_random_parity();
	return test_summary();
}