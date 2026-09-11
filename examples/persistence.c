/* persistence.c — file-backed store: writes survive sepal_close/reopen.
 *
 * Usage: ./persistence [path]   (default /tmp/sepal_example.db)
 * First run stores the vectors; later runs find them already there
 * (the store count is restored from the file on open). */

#include <stdio.h>

#include <ttypt/sepal.h>

int
main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : "/tmp/sepal_example.db";
	int err = 0;
	sepal_vecstore_t *vs = sepal_open(path, &err);
	if (!vs) {
		fprintf(stderr, "sepal_open(%s) failed: err=%d\n", path, err);
		return 1;
	}
	printf("open %s: %zu vectors already stored\n", path, sepal_n(vs));

	if (sepal_n(vs) == 0) {
		float v[64];
		for (unsigned r = 0; r < 20; r++) {
			for (unsigned i = 0; i < 64; i++)
				v[i] = (float)(((r * 131 + i * 17) % 200) - 100) / 100.0f;
			sepal_put(vs, (rec_ref_t)r, v, 64);
		}
		printf("stored 20 vectors\n");
	}

	float got[256];
	size_t d = sepal_get(vs, 0, got, sizeof(got) / sizeof(got[0]));
	printf("ref 0: dim=%zu full=%zu\n", d, sepal_full_dim(vs, 0));

	sepal_close(vs);   /* persists via qmap_save(); never qmap_close() */
	printf("closed (re-run to see the vectors survive)\n");
	return 0;
}
