#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct gguf_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;

namespace ced {

class Backend;

// All config is read from the GGUF (metadata-driven); nothing is hardcoded.
struct CedConfig {
    std::string arch;
    // transformer
    uint32_t embed_dim = 0, depth = 0, num_heads = 0, outputdim = 0;
    float    mlp_ratio = 4.0f;
    bool     qkv_bias = true;
    uint32_t patch_size = 16, patch_stride = 16, target_length = 0;
    std::string pooling = "mean";
    float    ln_eps_encoder = 1e-6f, ln_eps_head = 1e-5f, bn_eps = 1e-5f;
    // mel frontend
    uint32_t sample_rate = 16000, n_mels = 0, n_fft = 0, win_size = 0, hop_size = 0, n_freqs = 0;
    float    f_min = 0.0f, f_max = 8000.0f;
    bool     center = true;
    float    a2db_multiplier = 10.0f, a2db_amin = 1e-10f, a2db_top_db = 120.0f, a2db_ref = 1.0f;
    // labels (AudioSet ontology, index order)
    std::vector<std::string> labels;
};

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    bool load(const std::string& path);
    const CedConfig& config() const { return cfg_; }
    ggml_tensor* tensor(const std::string& name) const;  // nullptr if absent
    // Make every weight usable as a graph leaf on `backend`. CPU: zero-copy
    // (wraps the ctx mem buffer). GPU: one upload into a device buffer at load,
    // after which the host copy is released; tensor() then returns the device
    // tensors. Idempotent.
    bool realize_weights(const Backend& backend);
    // Host f32 copy of a small tensor the CPU side reads directly (mel window,
    // mel filterbank, init_bn stats). Valid after realize_weights, on any
    // device. nullptr if absent or not f32.
    const float* host_f32(const std::string& name) const;

private:
    CedConfig cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_context* dev_ctx_ = nullptr;          // no_alloc mirror of ctx_ (GPU path)
    ggml_backend_buffer_t weights_buf_ = nullptr;
    std::unordered_map<std::string, ggml_tensor*> tensors_;
    std::unordered_map<std::string, std::vector<float>> host_;
};

}  // namespace ced
