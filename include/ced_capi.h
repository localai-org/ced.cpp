#ifndef CED_CAPI_H
#define CED_CAPI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Flat C-API for ced.cpp — designed for dlopen / cgo / purego (LocalAI).
//
// All functions are extern "C" and never let a C++ exception cross the
// boundary. The model is loaded ONCE into an opaque `ced_ctx` and reused across
// classify calls (a single ctx is single-threaded: serialize calls or use one
// ctx per worker). Returned strings are malloc'd UTF-8 owned by the caller and
// released with ced_capi_free_string.
//
// The per-PCM entry points (ced_capi_classify_pcm*) take an arbitrary-length
// mono buffer, so a realtime/websocket consumer can call them on a sliding
// window for live recognition.

typedef struct ced_ctx ced_ctx;

// ABI version. Bump on any breaking change below. v1: initial classify API.
// The *_load_from_memory* functions were added later and do not change the version.
int ced_capi_abi_version(void);

// Load a GGUF model. Returns an owning context or NULL on failure (on failure,
// ced_capi_last_error(NULL) holds the message). Release with ced_capi_free.
ced_ctx* ced_capi_load(const char* gguf_path);

// Load a GGUF model from memory instead of a file. `data` points to the bytes of
// a complete GGUF file of `size` bytes. Nothing is read from disk and no
// temporary file is made, on any platform.
//
// Ownership: the buffer is only read during this call. The loader copies the
// tensor data into its own memory, so the caller may free or overwrite `data`
// as soon as the call returns, whatever the result. While the call runs, the
// memory use peaks at about the buffer size plus the model size.
//
// A truncated or corrupt buffer is rejected: the call returns NULL and
// ced_capi_last_error(NULL) holds the reason. It never reads outside
// [data, data + size). Same threading rules as ced_capi_load: the error string
// is thread-local and loads from several threads at once are safe.
// Release the context with ced_capi_free.
ced_ctx* ced_capi_load_from_memory(const void* data, size_t size);

// As ced_capi_load_from_memory, for a model stored inside a larger GGUF such as
// a bundle. Every metadata key and every tensor name of the model is stored as
// `<prefix><name>` (for example prefix "ced." gives "ced.ced.depth" and
// "ced.encoder.init_bn.weight"). Only the tensors under the prefix are copied,
// so the caller can pass the bytes of the whole bundle and no standalone copy of
// the component is needed. `prefix` must not be NULL or empty (use
// ced_capi_load_from_memory for a standalone model). Same ownership rule: the
// buffer may be freed after the call. NULL, with the reason in
// ced_capi_last_error(NULL), if the prefix matches no tensors.
ced_ctx* ced_capi_load_from_memory_prefixed(const void* data, size_t size, const char* prefix);

// Free a context from ced_capi_load. Safe on NULL.
void ced_capi_free(ced_ctx* ctx);

// Last error string. Pass the ctx whose call failed, or NULL for load-time
// errors. Never NULL; "" when there is no error. Borrowed (do not free).
const char* ced_capi_last_error(const ced_ctx* ctx);

// Model introspection.
int         ced_capi_num_classes(const ced_ctx* ctx);   // e.g. 527
const char* ced_capi_label(const ced_ctx* ctx, int index);  // borrowed, NULL if oob
int         ced_capi_sample_rate(const ced_ctx* ctx);   // e.g. 16000

// Classify mono float PCM (`samples`, length `n_samples`, range ~[-1,1]). If
// `sample_rate` differs from the model rate the audio is linearly resampled.
// Clips longer than the model window are chunked and averaged.
//
// Returns a malloc'd JSON array of the top-k tags, score-descending:
//   [{"index":23,"label":"Baby cry, infant cry","score":0.87}, ...]
// `top_k <= 0` returns all classes. Free with ced_capi_free_string. NULL on
// error (see ced_capi_last_error).
char* ced_capi_classify_pcm_json(ced_ctx* ctx, const float* samples, int n_samples,
                                 int sample_rate, int top_k);

// As above, reading a WAV file (any sample rate / channel count; downmixed to
// mono and resampled to the model rate).
char* ced_capi_classify_path_json(ced_ctx* ctx, const char* wav_path, int top_k);

// Struct-array variant (no JSON / no allocation of strings to free). Fills up to
// `max_tags` top tags into `out` (score-descending) and returns the count
// written, or -1 on error. `label` pointers are borrowed from the context and
// valid until ced_capi_free.
typedef struct {
    int         index;
    float       score;
    const char* label;
} ced_tag;

int ced_capi_classify_pcm(ced_ctx* ctx, const float* samples, int n_samples, int sample_rate,
                          ced_tag* out, int max_tags);

// All class scores for mono float PCM, in class-index order (no sorting, no
// allocation). Writes min(n_out, ced_capi_num_classes(ctx)) floats into `out`
// and returns that count, or -1 on error (see ced_capi_last_error). Resamples
// like ced_capi_classify_pcm when `sample_rate` differs from the model rate.
int ced_capi_classify_pcm_probs(ced_ctx* ctx, const float* samples, int n_samples,
                                int sample_rate, float* out, int n_out);

// Free a string returned by a *_json function. Safe on NULL.
void ced_capi_free_string(char* s);

#ifdef __cplusplus
}
#endif
#endif  // CED_CAPI_H
