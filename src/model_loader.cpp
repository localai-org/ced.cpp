#include "model_loader.hpp"

#include "ced_runner.hpp"
#include "gguf_check.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ced {

namespace {

// Reads typed keys from a gguf_context. Every accessor checks the stored type
// first: ggml aborts on a type or array/scalar mismatch, and a model that comes
// from a caller's memory buffer must not be able to do that. A mismatch is
// remembered in `bad` and reported once by the loader.
struct Kv {
    gguf_context* g;
    std::string prefix;
    std::string bad;

    int64_t find(const char* k, gguf_type want, bool array = false) {
        int64_t id = gguf_find_key(g, (prefix + k).c_str());
        if (id < 0) return -1;
        const bool is_arr = gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY;
        const gguf_type t = is_arr ? gguf_get_arr_type(g, id) : gguf_get_kv_type(g, id);
        if (is_arr != array || t != want) {
            if (bad.empty()) bad = "metadata key '" + prefix + k + "' has an unexpected type";
            return -1;
        }
        return id;
    }
    uint32_t u32(const char* k, uint32_t d = 0) {
        int64_t id = find(k, GGUF_TYPE_UINT32);
        return id < 0 ? d : gguf_get_val_u32(g, id);
    }
    float f32(const char* k, float d = 0) {
        int64_t id = find(k, GGUF_TYPE_FLOAT32);
        return id < 0 ? d : gguf_get_val_f32(g, id);
    }
    bool boolean(const char* k, bool d = false) {
        int64_t id = find(k, GGUF_TYPE_BOOL);
        return id < 0 ? d : gguf_get_val_bool(g, id);
    }
    std::string str(const char* k, const char* d = "") {
        int64_t id = find(k, GGUF_TYPE_STRING);
        return id < 0 ? std::string(d) : std::string(gguf_get_val_str(g, id));
    }
};

}  // namespace

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
    err_.clear();
    struct gguf_init_params p { /*no_alloc=*/false, /*ctx=*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_) {
        err_ = "cannot open or parse GGUF file: " + path;
        std::fprintf(stderr, "ced: gguf open failed: %s\n", path.c_str());
        return false;
    }
    return read_model("");
}

bool ModelLoader::load_from_memory(const void* data, size_t size, const std::string& prefix) {
    err_.clear();
    if (!data || size == 0) {
        err_ = "empty model buffer";
        return false;
    }
    if (!gguf_precheck(data, size, &err_)) return false;
    if (prefix.empty()) {
        // ggml copies the tensor data into ctx_; `data` is not used after this call.
        struct gguf_init_params p { /*no_alloc=*/false, /*ctx=*/&ctx_ };
        gguf_ = gguf_init_from_buffer(data, size, p);
        if (!gguf_) {
            err_ = "model buffer is not a valid GGUF file (truncated or corrupt)";
            return false;
        }
        return read_model("");
    }

    // Prefixed view: parse only the header and tensor table, then copy just the
    // tensors under `prefix` into a context of our own, under their stripped names.
    ggml_context* meta = nullptr;
    struct gguf_init_params p { /*no_alloc=*/true, /*ctx=*/&meta };
    gguf_ = gguf_init_from_buffer(data, size, p);
    if (!gguf_) {
        err_ = "model buffer is not a valid GGUF file (truncated or corrupt)";
        return false;
    }
    struct MetaGuard {
        ggml_context* c;
        ~MetaGuard() { if (c) ggml_free(c); }
    } guard{meta};

    const uint64_t data_off = gguf_get_data_offset(gguf_);
    if (data_off > size) {
        err_ = "model buffer is truncated (tensor data starts past the end)";
        return false;
    }
    const uint64_t avail = size - data_off;
    const int64_t nt = gguf_get_n_tensors(gguf_);
    std::vector<int64_t> ids;
    size_t need = 0;
    for (int64_t i = 0; i < nt; ++i) {
        const char* nm = gguf_get_tensor_name(gguf_, i);
        if (std::strncmp(nm, prefix.c_str(), prefix.size()) != 0) continue;
        const uint64_t off = gguf_get_tensor_offset(gguf_, i);
        const uint64_t nb = gguf_get_tensor_size(gguf_, i);
        if (off > avail || nb > avail - off) {
            err_ = std::string("model buffer is truncated (tensor ") + nm + " is out of range)";
            return false;
        }
        ggml_tensor* src = ggml_get_tensor(meta, nm);
        if (!src || ggml_nbytes(src) != nb) {
            err_ = std::string("tensor table is inconsistent for ") + nm;
            return false;
        }
        need += GGML_PAD((size_t)nb, GGML_MEM_ALIGN) + GGML_MEM_ALIGN + ggml_tensor_overhead();
        ids.push_back(i);
    }
    if (ids.empty()) {
        err_ = "no tensors under prefix '" + prefix + "' in the model buffer";
        return false;
    }
    struct ggml_init_params ip { need + 4096, nullptr, /*no_alloc=*/false };
    ctx_ = ggml_init(ip);
    if (!ctx_) {
        err_ = "out of memory";
        return false;
    }
    const uint8_t* base = static_cast<const uint8_t*>(data) + data_off;
    for (int64_t i : ids) {
        const char* nm = gguf_get_tensor_name(gguf_, i);
        ggml_tensor* src = ggml_get_tensor(meta, nm);
        ggml_tensor* t = ggml_new_tensor(ctx_, src->type, GGML_MAX_DIMS, src->ne);
        if (!t) {
            err_ = "out of memory";
            return false;
        }
        ggml_set_name(t, nm + prefix.size());
        std::memcpy(t->data, base + gguf_get_tensor_offset(gguf_, i), ggml_nbytes(t));
    }
    return read_model(prefix);
}

bool ModelLoader::read_model(const std::string& prefix) {
    Kv kv{gguf_, prefix, {}};
    cfg_.arch       = kv.str("ced.arch", "ced");
    cfg_.embed_dim  = kv.u32("ced.embed_dim");
    cfg_.depth      = kv.u32("ced.depth");
    cfg_.num_heads  = kv.u32("ced.num_heads");
    cfg_.outputdim  = kv.u32("ced.outputdim");
    cfg_.mlp_ratio  = kv.f32("ced.mlp_ratio", 4.0f);
    cfg_.qkv_bias   = kv.boolean("ced.qkv_bias", true);
    cfg_.patch_size = kv.u32("ced.patch_size", 16);
    cfg_.patch_stride = kv.u32("ced.patch_stride", 16);
    cfg_.target_length = kv.u32("ced.target_length");
    cfg_.pooling    = kv.str("ced.pooling", "mean");
    cfg_.ln_eps_encoder = kv.f32("ced.ln_eps_encoder", 1e-6f);
    cfg_.ln_eps_head    = kv.f32("ced.ln_eps_head", 1e-5f);
    cfg_.bn_eps         = kv.f32("ced.bn_eps", 1e-5f);
    cfg_.sample_rate = kv.u32("ced.sample_rate", 16000);
    cfg_.n_mels   = kv.u32("ced.n_mels");
    cfg_.n_fft    = kv.u32("ced.n_fft");
    cfg_.win_size = kv.u32("ced.win_size");
    cfg_.hop_size = kv.u32("ced.hop_size");
    cfg_.n_freqs  = kv.u32("ced.n_freqs");
    cfg_.f_min    = kv.f32("ced.f_min", 0.0f);
    cfg_.f_max    = kv.f32("ced.f_max", 8000.0f);
    cfg_.center   = kv.boolean("ced.center", true);
    cfg_.a2db_multiplier = kv.f32("ced.a2db_multiplier", 10.0f);
    cfg_.a2db_amin   = kv.f32("ced.a2db_amin", 1e-10f);
    cfg_.a2db_top_db = kv.f32("ced.a2db_top_db", 120.0f);
    cfg_.a2db_ref    = kv.f32("ced.a2db_ref", 1.0f);
    {
        int64_t id = kv.find("ced.labels", GGUF_TYPE_STRING, /*array=*/true);
        if (id >= 0) {
            size_t n = gguf_get_arr_n(gguf_, id);
            cfg_.labels.resize(n);
            for (size_t i = 0; i < n; ++i) cfg_.labels[i] = gguf_get_arr_str(gguf_, id, i);
        }
    }
    if (!kv.bad.empty()) {
        err_ = kv.bad;
        return false;
    }
    // Every tensor of the context (path loads: the whole file; prefixed loads:
    // the component, names already stripped).
    for (ggml_tensor* t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t))
        tensors_[ggml_get_name(t)] = t;
    if (!(cfg_.embed_dim > 0 && cfg_.depth > 0 && cfg_.outputdim > 0)) {
        err_ = "GGUF is not a ced model (missing ced.embed_dim, ced.depth or ced.outputdim)";
        return false;
    }
    return true;
}

ggml_tensor* ModelLoader::tensor(const std::string& n) const {
    auto it = tensors_.find(n);
    return it == tensors_.end() ? nullptr : it->second;
}

}  // namespace ced
