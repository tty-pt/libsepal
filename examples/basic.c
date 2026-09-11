/* basic.c — libsepal quick start: open, put, get, search. */

#include <stdio.h>

#include <ttypt/sepal.h>

int
main(void)
{
	sepal_vecstore_t *vs = sepal_open(NULL, NULL);
	if (!vs) {
		fprintf(stderr, "sepal_open failed\n");
		return 1;
	}

	/* three toy 4-dim vectors */
	float red[4] = { 1, 0, 0, 0 };
	float green[4] = { 0, 1, 0, 0 };
	float blue[4] = { 0, 0, 1, 0 };
	sepal_put(vs, 1, red, 4);
	sepal_put(vs, 2, green, 4);
	sepal_put(vs, 3, blue, 4);
	printf("stored %zu vectors\n", sepal_n(vs));

	/* query near red */
	float q[4] = { 0.9f, 0.1f, 0.0f, 0.0f };
	sepal_hit_t hits[2];
	size_t n = sepal_search(vs, q, 4, 2, 0.0f, 0, hits);
	for (size_t i = 0; i < n; i++)
		printf("hit ref=%llu score=%.4f\n",
		       (unsigned long long)hits[i].ref, hits[i].score);

	sepal_close(vs);
	return 0;
}
