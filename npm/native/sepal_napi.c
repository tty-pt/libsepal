#include <node_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include <ttypt/corm.h>
#include <ttypt/sepal.h>

/* sepalCosine(vecA, vecB) -> number */
static napi_value SepalCosine(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_type_error(env, NULL, "Expected 2 arguments: (vecA, vecB)");
        return NULL;
    }

    uint32_t lenA = 0, lenB = 0;
    napi_get_array_length(env, args[0], &lenA);
    napi_get_array_length(env, args[1], &lenB);

    if (lenA == 0 || lenA != lenB) {
        napi_value zero;
        napi_create_double(env, 0.0, &zero);
        return zero;
    }

    float *vA = (float *)malloc(lenA * sizeof(float));
    float *vB = (float *)malloc(lenB * sizeof(float));
    if (!vA || !vB) {
        if (vA) free(vA);
        if (vB) free(vB);
        napi_throw_error(env, NULL, "Out of memory");
        return NULL;
    }

    for (uint32_t i = 0; i < lenA; i++) {
        napi_value elA, elB;
        napi_get_element(env, args[0], i, &elA);
        napi_get_element(env, args[1], i, &elB);

        double dA = 0.0, dB = 0.0;
        napi_get_value_double(env, elA, &dA);
        napi_get_value_double(env, elB, &dB);

        vA[i] = (float)dA;
        vB[i] = (float)dB;
    }

    float sim = sepal_cosine(vA, vB, lenA);

    free(vA);
    free(vB);

    napi_value res;
    napi_create_double(env, (double)sim, &res);
    return res;
}

/* Sepal Store Wrapper */
typedef struct {
    sepal_vecstore_t *vs;
    bool is_open;
} SepalStoreWrapper;

static void SepalStoreFinalizer(napi_env env, void *finalize_data, void *finalize_hint) {
    (void)env;
    (void)finalize_hint;
    SepalStoreWrapper *wrap = (SepalStoreWrapper *)finalize_data;
    if (wrap) {
        if (wrap->is_open && wrap->vs) {
            sepal_close(wrap->vs);
            wrap->is_open = false;
        }
        free(wrap);
    }
}

static napi_value SepalOpen(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    char path_buf[256] = {0};
    if (argc >= 1) {
        size_t len = 0;
        napi_get_value_string_utf8(env, args[0], path_buf, sizeof(path_buf) - 1, &len);
    }

    int err = 0;
    sepal_vecstore_t *vs = sepal_open(path_buf[0] ? path_buf : NULL, &err);
    if (!vs || err != 0) {
        napi_throw_error(env, NULL, "Failed to open sepal vector store");
        return NULL;
    }

    SepalStoreWrapper *wrap = (SepalStoreWrapper *)calloc(1, sizeof(SepalStoreWrapper));
    wrap->vs = vs;
    wrap->is_open = true;

    napi_value res;
    napi_create_object(env, &res);
    napi_wrap(env, res, wrap, SepalStoreFinalizer, NULL, NULL);
    return res;
}

static napi_value SepalPut(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 3) {
        napi_throw_type_error(env, NULL, "Expected (store, ref, vectorArray)");
        return NULL;
    }

    SepalStoreWrapper *wrap = NULL;
    napi_unwrap(env, args[0], (void **)&wrap);
    if (!wrap || !wrap->is_open) {
        napi_throw_error(env, NULL, "Sepal store is closed or invalid");
        return NULL;
    }

    uint32_t ref = 0;
    napi_get_value_uint32(env, args[1], &ref);

    uint32_t dim = 0;
    napi_get_array_length(env, args[2], &dim);

    float *v = (float *)malloc(dim * sizeof(float));
    for (uint32_t i = 0; i < dim; i++) {
        napi_value el;
        napi_get_element(env, args[2], i, &el);
        double d = 0.0;
        napi_get_value_double(env, el, &d);
        v[i] = (float)d;
    }

    int rc = sepal_put(wrap->vs, ref, v, dim);
    free(v);

    napi_value res;
    napi_get_boolean(env, rc == 0, &res);
    return res;
}

static napi_value SepalSearch(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_type_error(env, NULL, "Expected (store, queryVector, optionalK, optionalMinSim)");
        return NULL;
    }

    SepalStoreWrapper *wrap = NULL;
    napi_unwrap(env, args[0], (void **)&wrap);
    if (!wrap || !wrap->is_open) {
        napi_throw_error(env, NULL, "Sepal store is closed or invalid");
        return NULL;
    }

    uint32_t dim = 0;
    napi_get_array_length(env, args[1], &dim);

    float *q = (float *)malloc(dim * sizeof(float));
    for (uint32_t i = 0; i < dim; i++) {
        napi_value el;
        napi_get_element(env, args[1], i, &el);
        double d = 0.0;
        napi_get_value_double(env, el, &d);
        q[i] = (float)d;
    }

    uint32_t k = 10;
    if (argc >= 3) napi_get_value_uint32(env, args[2], &k);

    double min_sim = 0.0;
    if (argc >= 4) napi_get_value_double(env, args[3], &min_sim);

    sepal_hit_t *hits = (sepal_hit_t *)malloc(k * sizeof(sepal_hit_t));
    size_t written = sepal_search(wrap->vs, q, dim, k, (float)min_sim, 0, hits);
    free(q);

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

static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"cosine", NULL, SepalCosine, NULL, NULL, NULL, napi_default, NULL},
        {"open", NULL, SepalOpen, NULL, NULL, NULL, napi_default, NULL},
        {"put", NULL, SepalPut, NULL, NULL, NULL, napi_default, NULL},
        {"search", NULL, SepalSearch, NULL, NULL, NULL, napi_default, NULL}
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
