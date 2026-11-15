#include <node_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include <ttypt/corm.h>
#include <ttypt/cormr.h>
#include <ttypt/sepal.h>

/* Helper: extract float vector with zero-copy fast path for Float32Array */
static bool extract_float_vec(napi_env env, napi_value val, const float **out_ptr, size_t *out_len, float **out_owned) {
    *out_ptr = NULL;
    *out_len = 0;
    *out_owned = NULL;

    bool is_ta = false;
    napi_is_typedarray(env, val, &is_ta);
    if (is_ta) {
        napi_typedarray_type ta_type;
        size_t elem_len = 0;
        void *data = NULL;
        napi_get_typedarray_info(env, val, &ta_type, &elem_len, &data, NULL, NULL);
        if (ta_type == napi_float32_array) {
            *out_ptr = (const float *)data;
            *out_len = elem_len;
            return true;
        }
    }

    uint32_t arr_len = 0;
    if (napi_get_array_length(env, val, &arr_len) != napi_ok) {
        return false;
    }
    if (arr_len == 0) {
        return true;
    }

    float *buf = (float *)malloc(arr_len * sizeof(float));
    if (!buf) return false;
    for (uint32_t i = 0; i < arr_len; i++) {
        napi_value el;
        napi_get_element(env, val, i, &el);
        double d = 0.0;
        napi_get_value_double(env, el, &d);
        buf[i] = (float)d;
    }
    *out_ptr = buf;
    *out_len = (size_t)arr_len;
    *out_owned = buf;
    return true;
}

/* sepalCosine(vecA, vecB) -> number */
static napi_value SepalCosine(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_type_error(env, NULL, "Expected 2 arguments: (vecA, vecB)");
        return NULL;
    }

    const float *vA = NULL, *vB = NULL;
    size_t lenA = 0, lenB = 0;
    float *ownA = NULL, *ownB = NULL;

    if (!extract_float_vec(env, args[0], &vA, &lenA, &ownA) ||
        !extract_float_vec(env, args[1], &vB, &lenB, &ownB)) {
        free(ownA);
        free(ownB);
        napi_throw_error(env, NULL, "Invalid float vector arguments");
        return NULL;
    }

    float sim = 0.0f;
    if (lenA > 0 && lenA == lenB) {
        sim = sepal_cosine(vA, vB, lenA);
    }

    free(ownA);
    free(ownB);

    napi_value res;
    napi_create_double(env, (double)sim, &res);
    return res;
}

/* sepalSketch(vec) -> BigUint64Array */
static napi_value SepalSketch(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 1) {
        napi_throw_type_error(env, NULL, "Expected 1 argument: (vec)");
        return NULL;
    }

    const float *v = NULL;
    size_t len = 0;
    float *own = NULL;
    if (!extract_float_vec(env, args[0], &v, &len, &own)) {
        napi_throw_error(env, NULL, "Invalid float vector argument");
        return NULL;
    }

    size_t nwords = (len + 63) / 64;
    void *ab_data = NULL;
    napi_value ab, ta;
    napi_create_arraybuffer(env, nwords * sizeof(uint64_t), &ab_data, &ab);
    if (nwords > 0 && ab_data) {
        memset(ab_data, 0, nwords * sizeof(uint64_t));
        sepal_sketch(v, len, (uint64_t *)ab_data, nwords);
    }
    free(own);

    napi_create_typedarray(env, napi_biguint64_array, nwords, ab, 0, &ta);
    return ta;
}

/* sepalHamming(sketchA, sketchB) -> number */
static napi_value SepalHamming(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_type_error(env, NULL, "Expected 2 arguments: (sketchA, sketchB)");
        return NULL;
    }

    bool is_taA = false, is_taB = false;
    napi_is_typedarray(env, args[0], &is_taA);
    napi_is_typedarray(env, args[1], &is_taB);
    if (!is_taA || !is_taB) {
        napi_throw_type_error(env, NULL, "Expected BigUint64Array arguments");
        return NULL;
    }

    napi_typedarray_type tA, tB;
    size_t lenA = 0, lenB = 0;
    void *dA = NULL, *dB = NULL;
    napi_get_typedarray_info(env, args[0], &tA, &lenA, &dA, NULL, NULL);
    napi_get_typedarray_info(env, args[1], &tB, &lenB, &dB, NULL, NULL);

    uint32_t dist = 0;
    if (lenA > 0 && lenA == lenB && dA && dB) {
        const uint64_t *wA = (const uint64_t *)dA;
        const uint64_t *wB = (const uint64_t *)dB;
        for (size_t i = 0; i < lenA; i++) {
            dist += (uint32_t)__builtin_popcountll(wA[i] ^ wB[i]);
        }
    }

    napi_value res;
    napi_create_uint32(env, dist, &res);
    return res;
}

/* Sepal Store Wrapper */
typedef struct {
    sepal_vecstore_t *vs;
    bool is_open;
    char *path_copy;
} SepalStoreWrapper;

static void SepalStoreFinalizer(napi_env env, void *finalize_data, void *finalize_hint) {
    (void)env;
    (void)finalize_hint;
    SepalStoreWrapper *wrap = (SepalStoreWrapper *)finalize_data;
    if (wrap) {
        if (wrap->is_open && wrap->vs) {
            sepal_close(wrap->vs);
            wrap->vs = NULL;
            wrap->is_open = false;
        }
        free(wrap->path_copy);
        free(wrap);
    }
}

static napi_value SepalOpen(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    char path_buf[512] = {0};
    if (argc >= 1) {
        size_t len = 0;
        napi_get_value_string_utf8(env, args[0], path_buf, sizeof(path_buf) - 1, &len);
    }

    SepalStoreWrapper *wrap = (SepalStoreWrapper *)calloc(1, sizeof(SepalStoreWrapper));
    if (!wrap) {
        napi_throw_error(env, NULL, "Out of memory");
        return NULL;
    }
    if (path_buf[0]) {
        wrap->path_copy = strdup(path_buf);
    }

    int err = 0;
    sepal_vecstore_t *vs = sepal_open(wrap->path_copy, &err);
    if (!vs || err != 0) {
        free(wrap->path_copy);
        free(wrap);
        napi_throw_error(env, NULL, "Failed to open sepal vector store");
        return NULL;
    }

    wrap->vs = vs;
    wrap->is_open = true;

    napi_value res;
    napi_create_object(env, &res);
    napi_wrap(env, res, wrap, SepalStoreFinalizer, NULL, NULL);
    return res;
}

static napi_value SepalClose(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc >= 1) {
        SepalStoreWrapper *wrap = NULL;
        napi_unwrap(env, args[0], (void **)&wrap);
        if (wrap && wrap->is_open && wrap->vs) {
            sepal_close(wrap->vs);
            wrap->vs = NULL;
            wrap->is_open = false;
        }
    }
    napi_value res;
    napi_get_boolean(env, true, &res);
    return res;
}

static napi_value SepalPut(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 3) {
        napi_throw_type_error(env, NULL, "Expected (store, ref, vector)");
        return NULL;
    }

    SepalStoreWrapper *wrap = NULL;
    napi_unwrap(env, args[0], (void **)&wrap);
    if (!wrap || !wrap->is_open || !wrap->vs) {
        napi_throw_error(env, NULL, "Sepal store is closed or invalid");
        return NULL;
    }

    uint32_t ref = 0;
    napi_get_value_uint32(env, args[1], &ref);

    const float *v = NULL;
    size_t dim = 0;
    float *own = NULL;
    if (!extract_float_vec(env, args[2], &v, &dim, &own) || dim == 0) {
        free(own);
        napi_throw_type_error(env, NULL, "Expected non-empty float vector");
        return NULL;
    }

    int rc = sepal_put(wrap->vs, (rec_ref_t)ref, v, dim);
    free(own);

    napi_value res;
    napi_get_boolean(env, rc == 0, &res);
    return res;
}

static napi_value SepalGet(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_type_error(env, NULL, "Expected (store, ref)");
        return NULL;
    }

    SepalStoreWrapper *wrap = NULL;
    napi_unwrap(env, args[0], (void **)&wrap);
    if (!wrap || !wrap->is_open || !wrap->vs) {
        napi_throw_error(env, NULL, "Sepal store is closed or invalid");
        return NULL;
    }

    uint32_t ref = 0;
    napi_get_value_uint32(env, args[1], &ref);

    size_t dim = sepal_dim(wrap->vs, (rec_ref_t)ref);
    if (dim == 0) {
        napi_value null_v;
        napi_get_null(env, &null_v);
        return null_v;
    }

    void *ab_data = NULL;
    napi_value ab, ta;
    napi_create_arraybuffer(env, dim * sizeof(float), &ab_data, &ab);
    size_t got = sepal_get(wrap->vs, (rec_ref_t)ref, (float *)ab_data, dim);
    if (got == 0) {
        napi_value null_v;
        napi_get_null(env, &null_v);
        return null_v;
    }

    napi_create_typedarray(env, napi_float32_array, got, ab, 0, &ta);
    return ta;
}

static napi_value SepalDel(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_type_error(env, NULL, "Expected (store, ref)");
        return NULL;
    }

    SepalStoreWrapper *wrap = NULL;
    napi_unwrap(env, args[0], (void **)&wrap);
    if (!wrap || !wrap->is_open || !wrap->vs) {
        napi_throw_error(env, NULL, "Sepal store is closed or invalid");
        return NULL;
    }

    uint32_t ref = 0;
    napi_get_value_uint32(env, args[1], &ref);

    int rc = sepal_del(wrap->vs, (rec_ref_t)ref);
    napi_value res;
    napi_get_boolean(env, rc == 0, &res);
    return res;
}

static napi_value SepalCount(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 1) {
        napi_throw_type_error(env, NULL, "Expected (store)");
        return NULL;
    }

    SepalStoreWrapper *wrap = NULL;
    napi_unwrap(env, args[0], (void **)&wrap);
    if (!wrap || !wrap->is_open || !wrap->vs) {
        napi_throw_error(env, NULL, "Sepal store is closed or invalid");
        return NULL;
    }

    size_t n = sepal_n(wrap->vs);
    napi_value res;
    napi_create_uint32(env, (uint32_t)n, &res);
    return res;
}

static napi_value SepalSearch(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_type_error(env, NULL, "Expected (store, queryVector[, k, minSim, m])");
        return NULL;
    }

    SepalStoreWrapper *wrap = NULL;
    napi_unwrap(env, args[0], (void **)&wrap);
    if (!wrap || !wrap->is_open || !wrap->vs) {
        napi_throw_error(env, NULL, "Sepal store is closed or invalid");
        return NULL;
    }

    const float *q = NULL;
    size_t dim = 0;
    float *own = NULL;
    if (!extract_float_vec(env, args[1], &q, &dim, &own) || dim == 0) {
        free(own);
        napi_value empty;
        napi_create_array_with_length(env, 0, &empty);
        return empty;
    }

    uint32_t k = 10;
    if (argc >= 3) napi_get_value_uint32(env, args[2], &k);
    if (k == 0) k = 10;

    double min_sim = -1.0;
    if (argc >= 4) napi_get_value_double(env, args[3], &min_sim);

    uint32_t m = 0;
    if (argc >= 5) napi_get_value_uint32(env, args[4], &m);

    sepal_hit_t *hits = (sepal_hit_t *)malloc(k * sizeof(sepal_hit_t));
    size_t written = sepal_search(wrap->vs, q, dim, k, (float)min_sim, m, hits);
    free(own);

    napi_value out_arr;
    napi_create_array_with_length(env, written, &out_arr);
    for (size_t i = 0; i < written; i++) {
        napi_value item;
        napi_create_object(env, &item);

        napi_value ref_val, score_val;
        napi_create_uint32(env, hits[i].ref, &ref_val);
        napi_create_double(env, (double)hits[i].score, &score_val);

        napi_set_named_property(env, item, "ref", ref_val);
        napi_set_named_property(env, item, "score", score_val);
        napi_set_element(env, out_arr, (uint32_t)i, item);
    }

    free(hits);
    return out_arr;
}

static napi_value SepalFillApprox(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_type_error(env, NULL, "Expected (store, queryVector[, m, minSim])");
        return NULL;
    }

    SepalStoreWrapper *wrap = NULL;
    napi_unwrap(env, args[0], (void **)&wrap);
    if (!wrap || !wrap->is_open || !wrap->vs) {
        napi_throw_error(env, NULL, "Sepal store is closed or invalid");
        return NULL;
    }

    const float *q = NULL;
    size_t dim = 0;
    float *own = NULL;
    if (!extract_float_vec(env, args[1], &q, &dim, &own) || dim == 0) {
        free(own);
        napi_value empty;
        napi_create_array_with_length(env, 0, &empty);
        return empty;
    }

    uint32_t m = 64;
    if (argc >= 3) napi_get_value_uint32(env, args[2], &m);

    double min_sim = -1.0;
    if (argc >= 4) napi_get_value_double(env, args[3], &min_sim);

    rec_set_t *cands = rec_set_new();
    sepal_fill_approx(wrap->vs, q, dim, (size_t)m, (float)min_sim, cands);
    free(own);

    size_t n = rec_set_count(cands);
    const rec_ref_t *refs = rec_set_at(cands);

    napi_value arr;
    napi_create_array_with_length(env, n, &arr);
    for (size_t i = 0; i < n; i++) {
        napi_value v;
        napi_create_uint32(env, (uint32_t)refs[i], &v);
        napi_set_element(env, arr, (uint32_t)i, v);
    }
    rec_set_free(cands);
    return arr;
}

static napi_value SepalRank(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 3) {
        napi_throw_type_error(env, NULL, "Expected (store, queryVector, ref[, minSim])");
        return NULL;
    }

    SepalStoreWrapper *wrap = NULL;
    napi_unwrap(env, args[0], (void **)&wrap);
    if (!wrap || !wrap->is_open || !wrap->vs) {
        napi_throw_error(env, NULL, "Sepal store is closed or invalid");
        return NULL;
    }

    const float *q = NULL;
    size_t dim = 0;
    float *own = NULL;
    if (!extract_float_vec(env, args[1], &q, &dim, &own) || dim == 0) {
        free(own);
        napi_value zero;
        napi_create_double(env, 0.0, &zero);
        return zero;
    }

    uint32_t ref = 0;
    napi_get_value_uint32(env, args[2], &ref);

    double min_sim = -1.0;
    if (argc >= 4) napi_get_value_double(env, args[3], &min_sim);

    struct sepal_rank_ctx rctx = { wrap->vs, q, dim, (float)min_sim };
    float score = 0.0f;
    if (sepal_rank(&rctx, (rec_ref_t)ref, &score) != 0) {
        score = 0.0f;
    }
    free(own);

    napi_value res;
    napi_create_double(env, (double)score, &res);
    return res;
}

static napi_value SepalStoreTyped(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 3) {
        napi_throw_type_error(env, NULL, "Expected (store, ref, uint8Array[, qtype])");
        return NULL;
    }

    SepalStoreWrapper *wrap = NULL;
    napi_unwrap(env, args[0], (void **)&wrap);
    if (!wrap || !wrap->is_open || !wrap->vs) {
        napi_throw_error(env, NULL, "Sepal store is closed or invalid");
        return NULL;
    }

    uint32_t ref = 0;
    napi_get_value_uint32(env, args[1], &ref);

    bool is_ta = false;
    napi_is_typedarray(env, args[2], &is_ta);
    if (!is_ta) {
        napi_throw_type_error(env, NULL, "Expected TypedArray for binary payload");
        return NULL;
    }

    napi_typedarray_type ta_type;
    size_t byte_len = 0;
    void *data = NULL;
    napi_get_typedarray_info(env, args[2], &ta_type, &byte_len, &data, NULL, NULL);

    uint32_t qtype = 0;
    if (argc >= 4) {
        napi_get_value_uint32(env, args[3], &qtype);
    }

    int rc = rec_axis_store_typed(wrap->vs, NULL, (rec_ref_t)ref, data, byte_len, qtype);
    napi_value res;
    napi_get_boolean(env, rc == 0, &res);
    return res;
}

static napi_value SepalConfigureEmbeddings(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    char url[512] = {0};
    char model[256] = {0};
    char key[512] = {0};

    if (argc >= 1) {
        size_t l = 0;
        napi_get_value_string_utf8(env, args[0], url, sizeof(url) - 1, &l);
    }
    if (argc >= 2) {
        size_t l = 0;
        napi_get_value_string_utf8(env, args[1], model, sizeof(model) - 1, &l);
    }
    if (argc >= 3) {
        size_t l = 0;
        napi_get_value_string_utf8(env, args[2], key, sizeof(key) - 1, &l);
    }

    int rc = sepal_configure_embeddings(
        url[0] ? url : NULL,
        model[0] ? model : NULL,
        key[0] ? key : NULL
    );
    napi_value res;
    napi_get_boolean(env, rc == 0, &res);
    return res;
}

static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"cosine", NULL, SepalCosine, NULL, NULL, NULL, napi_default, NULL},
        {"sketch", NULL, SepalSketch, NULL, NULL, NULL, napi_default, NULL},
        {"hamming", NULL, SepalHamming, NULL, NULL, NULL, napi_default, NULL},
        {"open", NULL, SepalOpen, NULL, NULL, NULL, napi_default, NULL},
        {"close", NULL, SepalClose, NULL, NULL, NULL, napi_default, NULL},
        {"put", NULL, SepalPut, NULL, NULL, NULL, napi_default, NULL},
        {"get", NULL, SepalGet, NULL, NULL, NULL, napi_default, NULL},
        {"del", NULL, SepalDel, NULL, NULL, NULL, napi_default, NULL},
        {"count", NULL, SepalCount, NULL, NULL, NULL, napi_default, NULL},
        {"search", NULL, SepalSearch, NULL, NULL, NULL, napi_default, NULL},
        {"fillApprox", NULL, SepalFillApprox, NULL, NULL, NULL, napi_default, NULL},
        {"rank", NULL, SepalRank, NULL, NULL, NULL, napi_default, NULL},
        {"storeTyped", NULL, SepalStoreTyped, NULL, NULL, NULL, napi_default, NULL},
        {"configureEmbeddings", NULL, SepalConfigureEmbeddings, NULL, NULL, NULL, napi_default, NULL}
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
