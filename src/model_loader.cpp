#include "model_loader.hpp"

#include "ced_runner.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>

namespace ced {

static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d = 0) {
    int64_t id = gguf_find_key(g, k);
    return id < 0 ? d : (uint32_t)gguf_get_val_u32(g, id);
}
static float kv_f32(gguf_context* g, const char* k, float d = 0) {
    int64_t id = gguf_find_key(g, k);
    return id < 0 ? d : gguf_get_val_f32(g, id);
}
static bool kv_bool(gguf_context* g, const char* k, bool d = false) {
    int64_t id = gguf_find_key(g, k);
    return id < 0 ? d : gguf_get_val_bool(g, id);
}
static std::string kv_str(gguf_context* g, const char* k, const char* d = "") {
    int64_t id = gguf_find_key(g, k);
    return id < 0 ? std::string(d) : std::string(gguf_get_val_str(g, id));
}

ModelLoader::~ModelLoader() {
    if (weights_buf_) ggml_backend_buffer_free(weights_buf_);
    if (gguf_) gguf_free(gguf_);
    if (dev_ctx_) ggml_free(dev_ctx_);
    if (ctx_) ggml_free(ctx_);
}

// Tensors read on the host (the mel frontend and the init_bn fold), kept as
// host copies so they stay readable once the weights move to a device.
static bool host_side(const std::string& n) {
    return n.rfind("ced.mel_", 0) == 0 || n.rfind("encoder.init_bn.", 0) == 0;
}

bool ModelLoader::realize_weights(const Backend& backend) {
    if (weights_buf_) return true;  // idempotent
    if (!ctx_ || !backend.ok()) return false;

    for (auto& kv : tensors_) {
        ggml_tensor* t = kv.second;
        if (!host_side(kv.first) || t->type != GGML_TYPE_F32) continue;
        const float* p = (const float*)t->data;
        host_[kv.first].assign(p, p + ggml_nelements(t));
    }

    if (backend.is_cpu()) {
        // The GGUF was loaded no_alloc=false, so every tensor's data lives in
        // one contiguous ctx mem buffer. Wrap that exact memory as a CPU
        // backend buffer (zero-copy) and point every tensor's ->buffer at it;
        // graphs then reference the loader's tensors directly as leaves.
        void* base = ggml_get_mem_buffer(ctx_);
        size_t size = ggml_get_mem_size(ctx_);
        weights_buf_ = ggml_backend_cpu_buffer_from_ptr(base, size);
        if (!weights_buf_) return false;
        for (auto& kv : tensors_) kv.second->buffer = weights_buf_;
        return true;
    }

    // Device: mirror every tensor into a no_alloc context, allocate them all in
    // one device buffer, upload once, then drop the host copy.
    struct ggml_init_params ip { ggml_tensor_overhead() * (tensors_.size() + 1), nullptr,
                                 /*no_alloc=*/true };
    dev_ctx_ = ggml_init(ip);
    if (!dev_ctx_) return false;
    std::unordered_map<std::string, ggml_tensor*> dev;
    for (auto& kv : tensors_) {
        ggml_tensor* d = ggml_dup_tensor(dev_ctx_, kv.second);
        ggml_set_name(d, kv.first.c_str());
        dev[kv.first] = d;
    }
    weights_buf_ = ggml_backend_alloc_ctx_tensors(dev_ctx_, backend.handle());
    if (!weights_buf_) {
        std::fprintf(stderr, "ced: device weight allocation failed\n");
        return false;
    }
    ggml_backend_buffer_set_usage(weights_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto& kv : tensors_)
        ggml_backend_tensor_set(dev[kv.first], kv.second->data, 0, ggml_nbytes(kv.second));
    tensors_ = std::move(dev);
    ggml_free(ctx_);
    ctx_ = nullptr;
    return true;
}

const float* ModelLoader::host_f32(const std::string& n) const {
    auto it = host_.find(n);
    return it == host_.end() ? nullptr : it->second.data();
}

bool ModelLoader::load(const std::string& path) {
    struct gguf_init_params p { /*no_alloc=*/false, /*ctx=*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_) {
        std::fprintf(stderr, "ced: gguf open failed: %s\n", path.c_str());
        return false;
    }
    cfg_.arch       = kv_str(gguf_, "ced.arch", "ced");
    cfg_.embed_dim  = kv_u32(gguf_, "ced.embed_dim");
    cfg_.depth      = kv_u32(gguf_, "ced.depth");
    cfg_.num_heads  = kv_u32(gguf_, "ced.num_heads");
    cfg_.outputdim  = kv_u32(gguf_, "ced.outputdim");
    cfg_.mlp_ratio  = kv_f32(gguf_, "ced.mlp_ratio", 4.0f);
    cfg_.qkv_bias   = kv_bool(gguf_, "ced.qkv_bias", true);
    cfg_.patch_size = kv_u32(gguf_, "ced.patch_size", 16);
    cfg_.patch_stride = kv_u32(gguf_, "ced.patch_stride", 16);
    cfg_.target_length = kv_u32(gguf_, "ced.target_length");
    cfg_.pooling    = kv_str(gguf_, "ced.pooling", "mean");
    cfg_.ln_eps_encoder = kv_f32(gguf_, "ced.ln_eps_encoder", 1e-6f);
    cfg_.ln_eps_head    = kv_f32(gguf_, "ced.ln_eps_head", 1e-5f);
    cfg_.bn_eps         = kv_f32(gguf_, "ced.bn_eps", 1e-5f);
    cfg_.sample_rate = kv_u32(gguf_, "ced.sample_rate", 16000);
    cfg_.n_mels   = kv_u32(gguf_, "ced.n_mels");
    cfg_.n_fft    = kv_u32(gguf_, "ced.n_fft");
    cfg_.win_size = kv_u32(gguf_, "ced.win_size");
    cfg_.hop_size = kv_u32(gguf_, "ced.hop_size");
    cfg_.n_freqs  = kv_u32(gguf_, "ced.n_freqs");
    cfg_.f_min    = kv_f32(gguf_, "ced.f_min", 0.0f);
    cfg_.f_max    = kv_f32(gguf_, "ced.f_max", 8000.0f);
    cfg_.center   = kv_bool(gguf_, "ced.center", true);
    cfg_.a2db_multiplier = kv_f32(gguf_, "ced.a2db_multiplier", 10.0f);
    cfg_.a2db_amin   = kv_f32(gguf_, "ced.a2db_amin", 1e-10f);
    cfg_.a2db_top_db = kv_f32(gguf_, "ced.a2db_top_db", 120.0f);
    cfg_.a2db_ref    = kv_f32(gguf_, "ced.a2db_ref", 1.0f);
    {
        int64_t id = gguf_find_key(gguf_, "ced.labels");
        if (id >= 0 && gguf_get_arr_type(gguf_, id) == GGUF_TYPE_STRING) {
            size_t n = gguf_get_arr_n(gguf_, id);
            cfg_.labels.resize(n);
            for (size_t i = 0; i < n; ++i) cfg_.labels[i] = gguf_get_arr_str(gguf_, id, i);
        }
    }
    const int64_t nt = gguf_get_n_tensors(gguf_);
    for (int64_t i = 0; i < nt; ++i) {
        const char* nm = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm);
        if (t) tensors_[nm] = t;
    }
    return cfg_.embed_dim > 0 && cfg_.depth > 0 && cfg_.outputdim > 0;
}

ggml_tensor* ModelLoader::tensor(const std::string& n) const {
    auto it = tensors_.find(n);
    return it == tensors_.end() ? nullptr : it->second;
}

}  // namespace ced
