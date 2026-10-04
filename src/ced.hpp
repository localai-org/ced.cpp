#pragma once
#include <memory>
#include <string>
#include <vector>

#include "ced_runner.hpp"
#include "model_loader.hpp"

namespace ced {

class Ced {
public:
    bool load(const std::string& path);
    // Load from a GGUF in memory (see ModelLoader::load_from_memory). `data` is
    // not needed after the call returns. `prefix` selects a model inside a larger
    // GGUF; leave it empty for a standalone model.
    bool load_from_memory(const void* data, size_t size, const std::string& prefix = "");
    // Why the last load failed ("" if it did not).
    const std::string& load_error() const { return err_; }
    const CedConfig& config() const { return loader_.config(); }
    // Compute device the model runs on ("cpu", "CUDA0", "Vulkan0", ...).
    const std::string& device_name() const { return backend_->device_name(); }

    // Parity entry point: run the 12 ViT blocks + final norm + mean-pool head
    // from precomputed patch tokens. `tokens` is n_tokens * embed_dim, token-
    // major (token t, channel c at t*embed_dim + c) — matching the baseline
    // `tokens_in` dump. Fills:
    //   enc_norm: n_tokens * embed_dim (post final LayerNorm)
    //   logits:   outputdim            (pre-sigmoid)
    //   probs:    outputdim            (post-sigmoid)
    bool forward_from_tokens(const std::vector<float>& tokens, int n_tokens,
                             std::vector<float>& enc_norm,
                             std::vector<float>& logits,
                             std::vector<float>& probs, int n_threads = 4);

    // Mel frontend: waveform -> input_values [n_mels * T] (mel-major), T frames.
    bool mel_frontend(const std::vector<float>& wav, std::vector<float>& input_values,
                      int& T);

    // init_bn + patch_embed + positional + flatten: input_values [n_mels*T] ->
    // tokens [n_tokens * embed_dim] (token-major). Also exposes intermediates
    // (init_bn_out [n_mels*T], patch_embed/pos_out [embed_dim*OH*OW]) for parity.
    bool embed_from_input_values(const std::vector<float>& input_values, int T,
                                 std::vector<float>& init_bn_out,
                                 std::vector<float>& patch_embed,
                                 std::vector<float>& pos_out, std::vector<float>& tokens,
                                 int& n_tokens, int n_threads = 4);

    // End-to-end: waveform -> logits/probs over the 527 AudioSet classes. Each
    // target_length chunk runs as ONE graph (embed + blocks + head).
    bool classify(const std::vector<float>& wav, std::vector<float>& logits,
                  std::vector<float>& probs, int n_threads = 4);

private:
    bool finish_load();
    struct Embed;
    struct Head;
    Embed build_embed(ggml_context* ctx, std::vector<GraphInput>& inputs,
                      const std::vector<float>& input_values, int T) const;
    Head build_blocks(ggml_context* ctx, ggml_tensor* tokens, int n_tokens) const;

    ModelLoader loader_;
    std::string err_;
    std::unique_ptr<Backend> backend_;
    // init_bn (BatchNorm2d, eval) folded into a per-mel scale/shift at load.
    std::vector<float> bn_scale_, bn_shift_;
};

}  // namespace ced
