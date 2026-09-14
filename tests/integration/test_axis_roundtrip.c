/*
 * test_axis_roundtrip.c — 2A-5: file-backed store, cross-process round-trip.
 *
 * Self-exec'ing harness: the orchestrator (no argv) fork+exec's itself one
 * phase at a time — seed, verify1, unstore, verify2. Every phase is a fresh
 * process, so "reopen" really reads what the previous phase's qmap_save()
 * (plus libqmap's exit destructor) flushed to disk; nothing is ever
 * qmap_close'd (no-close invariant). Proves the sepal blob store round-trips
 * through its documented file-backed lifecycle:
 *
 *   seed     → rec_axis_store(1, "0.5,-0.25,0.75") and (2, "1.0,2.0,3.0")
 *   verify1  → reopen → sepal_search finds ref 1 (cosine 1.0), ref 2 stored
 *   unstore  → reopen → rec_axis_unstore(1)  (ref 2 untouched: isolation)
 *   verify2  → reopen → ref 1 gone from search, ref 2 still found, readback(1)
 *              NULL → absent after reopen
 */

#include "../test_common.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>

#include <ttypt/qmap.h>

#define DB_PATH "/tmp/test_sepal_roundtrip.db"

static float vec_ref1[] = { 0.5f, -0.25f, 0.75f };
static float vec_ref2[] = { 1.0f, 2.0f, 3.0f };

/* one worker hit in a search for vec_ref1 */
static int
has_hit(const sepal_hit_t *hits, size_t n, rec_ref_t want, float min_score)
{
	size_t i;

	for (i = 0; i < n; i++)
		if (hits[i].ref == want && hits[i].score >= min_score)
			return 1;
	return 0;
}

static int
phase_seed(void)
{
	sepal_vecstore_t *vs = rec_axis_open(DB_PATH);

	if (!vs)
		return 1;
	if (rec_axis_store(vs, NULL, 1, "0.5,-0.25,0.75") != 0)
		return 1;
	if (rec_axis_store(vs, NULL, 2, "1.0,2.0,3.0") != 0)
		return 1;
	if (sepal_n(vs) != 2)
		return 1;
	qmap_save();
	return 0;
}

static int
phase_verify1(void)
{
	sepal_vecstore_t *vs = rec_axis_open(DB_PATH);
	sepal_hit_t hits[8];
	float got[3];
	size_t n;

	if (!vs)
		return 1;
	if (sepal_n(vs) != 2)
		return 1;
	n = sepal_search(vs, vec_ref1, 3, 8, 0.1f, 8, hits);
	if (!has_hit(hits, n, 1, 0.99f))
		return 1;                      /* reopened store answers the query */
	if (sepal_get(vs, 2, got, 3) != 3)
		return 1;                      /* ref 2 rehydrated too */
	if (got[1] != 2.0f)
		return 1;
	return 0;
}

static int
phase_unstore(void)
{
	sepal_vecstore_t *vs = rec_axis_open(DB_PATH);

	if (!vs)
		return 1;
	if (rec_axis_unstore(vs, 1) != 0)
		return 1;
	if (sepal_n(vs) != 1)              /* isolation: ref 2 kept */
		return 1;
	qmap_save();
	return 0;
}

static int
phase_verify2(void)
{
	sepal_vecstore_t *vs = rec_axis_open(DB_PATH);
	sepal_hit_t hits[8];
	char *blob = (char *)0x1;
	size_t n = 1;

	if (!vs)
		return 1;
	if (sepal_n(vs) != 1)
		return 1;
	n = sepal_search(vs, vec_ref1, 3, 8, 0.1f, 8, hits);
	if (has_hit(hits, n, 1, 0.99f))
		return 1;                      /* ref 1 gone after reopen */
	n = sepal_search(vs, vec_ref2, 3, 8, 0.1f, 8, hits);
	if (!has_hit(hits, n, 2, 0.99f))
		return 1;                      /* ref 2 survived the reopen */
	if (rec_axis_readback(vs, 1, &blob, &n) != 0 || blob != NULL || n != 0)
		return 1;                      /* absent after reopen */
	if (sepal_dim(vs, 1) != 0)
		return 1;
	return 0;
}

static void
unlink_stale(void)
{
	unlink(DB_PATH);
}

static int
run_phase(const char *prog, const char *mode)
{
	pid_t pid;
	int st = 1;

	pid = fork();
	if (pid < 0)
		return 1;
	if (pid == 0) {
		execl(prog, prog, mode, (char *)NULL);
		_exit(127);
	}
	if (waitpid(pid, &st, 0) < 0)
		return 1;
	return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}

int
main(int argc, char **argv)
{
	const char *mode = argc > 1 ? argv[1] : NULL;

	if (mode) {
		if (strcmp(mode, "seed") == 0)
			return phase_seed();
		if (strcmp(mode, "verify1") == 0)
			return phase_verify1();
		if (strcmp(mode, "unstore") == 0)
			return phase_unstore();
		if (strcmp(mode, "verify2") == 0)
			return phase_verify2();
		fprintf(stderr, "unknown mode: %s\n", mode);
		return 2;
	}

	/* orchestrator */
	printf("=== 2A-5 file-backed cross-process round-trip ===\n");
	unlink_stale();
	if (run_phase(argv[0], "seed") != 0) {
		printf("seed phase failed\n");
		return 1;
	}
	if (run_phase(argv[0], "verify1") != 0) {
		printf("reopen → query finds ref: FAILED\n");
		return 1;
	}
	printf("reopen → query finds ref: ok\n");
	if (run_phase(argv[0], "unstore") != 0) {
		printf("unstore phase failed\n");
		return 1;
	}
	if (run_phase(argv[0], "verify2") != 0) {
		printf("reopen → absent: FAILED\n");
		return 1;
	}
	printf("reopen → absent after unstore: ok\n");
	unlink_stale();
	printf("ALL ROUND-TRIP PHASES PASSED\n");
	return 0;
}