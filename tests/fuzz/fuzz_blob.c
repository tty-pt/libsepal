/* fuzz_blob.c — standalone blob parser fuzzer (deterministic, no libFuzzer).
 * Runs N iterations of adversarial buffers; any crash is a bug. */

#include "../test_common.h"

#ifndef FUZZ_ITERS
#define FUZZ_ITERS 50000
#endif

int
main(void)
{
	sepal_rng_t r = { 91 };
	uint8_t buf[1600];
	unsigned refused = 0, accepted = 0;

	for (unsigned i = 0; i < FUZZ_ITERS; i++) {
		size_t len = (size_t)(rng_next(&r) % sizeof(buf));
		for (size_t j = 0; j < len; j++)
			buf[j] = (uint8_t)(rng_next(&r) & 0xFF);
		switch (i % 4) {
		case 0:
			if (len >= 4) { buf[0]='V'; buf[1]='E'; buf[2]='C'; buf[3]='1'; }
			break;
		case 1:
			if (len >= 28) {
				/* structurally plausible-but-valid fields: small dim,
				 * full aligned up to the sketch's word multiple */
				uint16_t ver = 1;
				size_t dcap = len / 12 > 256 ? 256 : len / 12;
				uint16_t dim = (uint16_t)(1 + rng_next(&r) % dcap);
				uint16_t full = (uint16_t)(((dim + 63) / 64) * 64);
				uint16_t w = (uint16_t)(full / 64);
				if (16 + (size_t)8 * w + (size_t)4 * dim <= len) {
					buf[0] = 'V'; buf[1] = 'E'; buf[2] = 'C'; buf[3] = '1';
					memcpy(buf + 4, &ver, 2);
					memcpy(buf + 6, &dim, 2);
					memcpy(buf + 8, &full, 2);
					memcpy(buf + 10, &w, 2);
				}
			}
			break;
		default:
			break;
		}
		size_t l = sepal_blob_len(buf, len);
		sepal_blob_hdr_t h;
		int hr = sepal_blob_hdr(buf, len, &h);
		if (l != 0) {
			accepted++;
			if (hr != 0 || l > len || l < 16)
				return 1;   /* accepted but inconsistent => bug */
		} else {
			refused++;
			if (hr == 0)
				return 2;   /* refused but header parsed => bug */
		}
	}

	printf("fuzz_blob: %u iters, %u refused, %u accepted\n",
	       FUZZ_ITERS, refused, accepted);
	return 0;
}