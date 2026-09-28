#include "ced.hpp"

#include "mel.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace ced {

struct Ced::Embed {
    ggml_tensor* bn = nullptr;    // init_bn output   [T, n_mels]
    ggml_tensor* conv = nullptr;  // patch_embed      [OW, OH, OC]
    ggml_tensor* pos = nullptr;   // + positional     [OW, OH, OC]
    ggml_tensor* tokens = nullptr;  //                [OC, OW*OH]
};

struct Ced::Head {
    ggml_tensor* enc = nullptr;     // post final LayerNorm [C, N]
    ggml_tensor* logits = nullptr;  // [outputdim, 1]
    ggml_tensor* probs = nullptr;   // [outputdim, 1]
};

bool Ced::load(const std::string& path) {
    if (!loader_.load(path)) return false;
    backend_ = std::make_unique<Backend>();
    if (!backend_->ok() || !loader_.realize_weights(*backend_)) return false;

    const CedConfig& c = loader_.config();
    const float* bw = loader_.host_f32("encoder.init_bn.weight");
    const float* bb = loader_.host_f32("encoder.init_bn.bias");
    const float* bm = loader_.host_f32("encoder.init_bn.running_mean");
    const float* bv = loader_.host_f32("encoder.init_bn.running_var");
    if (!bw || !bb || !bm || !bv) {
        std::fprintf(stderr, "ced: missing init_bn tensors\n");
        return false;
    }
    bn_scale_.resize(c.n_mels);
    bn_shift_.resize(c.n_mels);
    for (uint32_t m = 0; m < c.n_mels; ++m) {
        bn_scale_[m] = bw[m] / std::sqrt(bv[m] + c.bn_eps);
        bn_shift_[m] = bb[m] - bm[m] * bn_scale_[m];
    }
    return true;
}

// LayerNorm over ne[0] (channel dim): y = norm(x, eps) * w + b. w/b are [C] and
// broadcast over the token dim.
static ggml_tensor* layernorm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                              ggml_tensor* b, float eps) {
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    y = ggml_add(ctx, y, b);
    return y;
}

// init_bn + patch_embed + positional + flatten: input_values [n_mels*T] ->
// tokens [C, n_tokens].
Ced::Embed Ced::build_embed(ggml_context* ctx, std::vector<GraphInput>& inputs,
                            const std::vector<float>& input_values, int T) const {
    const CedConfig& c = loader_.config();
    const int n_mels = (int)c.n_mels;
    const ModelLoader& L = loader_;
    auto W = [&](const std::string& n) { return L.tensor(n); };
    Embed e;

    // input_values leaf ne=[T, n_mels] (element (t,m) at t + m*T == mel-major flat)
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, n_mels);
    ggml_set_input(x);
    inputs.push_back({x, input_values.data(), input_values.size() * sizeof(float)});

    ggml_tensor* scaleT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_mels);
    ggml_set_input(scaleT);
    inputs.push_back({scaleT, bn_scale_.data(), bn_scale_.size() * sizeof(float)});
    ggml_tensor* shiftT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_mels);
    ggml_set_input(shiftT);
    inputs.push_back({shiftT, bn_shift_.data(), bn_shift_.size() * sizeof(float)});

    e.bn = ggml_add(ctx, ggml_mul(ctx, x, scaleT), shiftT);  // [T,n_mels]

    // patch_embed conv2d: kernel [KW,KH,IC,OC]=[16,16,1,768], input [W=T,H=mels,1,1]
    ggml_tensor* bn4 = ggml_reshape_4d(ctx, e.bn, T, n_mels, 1, 1);
    ggml_tensor* conv = ggml_conv_2d(ctx, W("encoder.patch_embed.proj.weight"), bn4,
                                     c.patch_stride, c.patch_stride, 0, 0, 1, 1);  // [OW,OH,OC,1]
    ggml_tensor* pbias =
        ggml_reshape_3d(ctx, W("encoder.patch_embed.proj.bias"), 1, 1, c.embed_dim);
    e.conv = ggml_add(ctx, conv, pbias);

    const int OW = (int)e.conv->ne[0], OH = (int)e.conv->ne[1];

    // positional. time_pos_embed is stored at the full target time grid
    // (ne=[63,1,OC]); the reference slices it to the actual time grid
    // (time_pos_embed[..., :OW]). For OW < 63 (any clip shorter than
    // target_length, or the padded last chunk) we must slice too, else the
    // broadcast add fails ggml_can_repeat. freq_pos_embed (ne=[1,OH,OC])
    // already broadcasts over OW, so it needs no slice.
    ggml_tensor* tpe = W("encoder.time_pos_embed");
    ggml_tensor* tpe_v = tpe;
    if (OW != (int)tpe->ne[0])
        tpe_v = ggml_cont(ctx, ggml_view_3d(ctx, tpe, OW, tpe->ne[1], tpe->ne[2],
                                            tpe->nb[1], tpe->nb[2], 0));
    e.pos = ggml_add(ctx, e.conv, tpe_v);
    e.pos = ggml_add(ctx, e.pos, W("encoder.freq_pos_embed"));

    // flatten freq-major to tokens: [OW,OH,OC] -> [OC,OW,OH] -> [OC, OW*OH]
    ggml_tensor* perm = ggml_cont(ctx, ggml_permute(ctx, e.pos, 1, 2, 0, 3));  // [OC,OW,OH]
    e.tokens = ggml_reshape_2d(ctx, perm, c.embed_dim, OW * OH);
    return e;
}

// 12 ViT blocks + final norm + mean-pool head over tokens [C, N].
Ced::Head Ced::build_blocks(ggml_context* ctx, ggml_tensor* x, int N) const {
    const CedConfig& c = loader_.config();
    const int C = (int)c.embed_dim;
    const int H = (int)c.num_heads;
    const int D = C / H;
    const float eps_e = c.ln_eps_encoder;
    const float eps_h = c.ln_eps_head;
    const ModelLoader& L = loader_;
    auto W = [&](const std::string& n) -> ggml_tensor* { return L.tensor(n); };
    auto blk = [&](int i, const char* suffix) -> ggml_tensor* {
        return L.tensor("encoder.blocks." + std::to_string(i) + "." + suffix);
    };

    const float scale = 1.0f / std::sqrt((float)D);
    for (int i = 0; i < (int)c.depth; ++i) {
        // --- attention ---
        ggml_tensor* n1 = layernorm(ctx, x, blk(i, "norm1.weight"),
                                    blk(i, "norm1.bias"), eps_e);
        ggml_tensor* qkv = ggml_mul_mat(ctx, blk(i, "attn.qkv.weight"), n1);  // [3C,N]
        qkv = ggml_add(ctx, qkv, blk(i, "attn.qkv.bias"));

        ggml_tensor* q = ggml_cont(
            ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1], 0 * (size_t)C * sizeof(float)));
        ggml_tensor* k = ggml_cont(
            ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1], 1 * (size_t)C * sizeof(float)));
        ggml_tensor* v = ggml_cont(
            ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1], 2 * (size_t)C * sizeof(float)));

        q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, D, H, N), 0, 2, 1, 3);  // [D,N,H]
        k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, D, H, N), 0, 2, 1, 3);  // [D,N,H]
        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                            // [N,N,H]
        kq = ggml_scale(ctx, kq, scale);
        kq = ggml_soft_max(ctx, kq);
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, D, H, N),
                                        1, 2, 0, 3));                          // [N,D,H]
        ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);                          // [D,N,H]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                             // [D,H,N]
        ggml_tensor* attn = ggml_cont_2d(ctx, kqv, C, N);
        attn = ggml_mul_mat(ctx, blk(i, "attn.proj.weight"), attn);
        attn = ggml_add(ctx, attn, blk(i, "attn.proj.bias"));
        x = ggml_add(ctx, x, attn);

        // --- mlp ---
        ggml_tensor* n2 = layernorm(ctx, x, blk(i, "norm2.weight"),
                                    blk(i, "norm2.bias"), eps_e);
        ggml_tensor* h = ggml_mul_mat(ctx, blk(i, "mlp.fc1.weight"), n2);
        h = ggml_add(ctx, h, blk(i, "mlp.fc1.bias"));
        h = ggml_gelu_erf(ctx, h);
        h = ggml_mul_mat(ctx, blk(i, "mlp.fc2.weight"), h);
        h = ggml_add(ctx, h, blk(i, "mlp.fc2.bias"));
        x = ggml_add(ctx, x, h);
    }

    Head out;
    // final encoder norm
    out.enc = layernorm(ctx, x, W("encoder.norm.weight"), W("encoder.norm.bias"), eps_e);

    // mean pool over tokens: [C,N] -> [C,1]
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, out.enc));  // [N,C]
    ggml_tensor* pooled = ggml_sum_rows(ctx, xt);                    // [1,C]
    pooled = ggml_scale(ctx, pooled, 1.0f / (float)N);
    pooled = ggml_reshape_2d(ctx, pooled, C, 1);                     // [C,1]

    // head: LayerNorm(eps_head) -> Linear(C->outputdim) -> sigmoid
    ggml_tensor* hn = ggml_norm(ctx, pooled, eps_h);
    hn = ggml_mul(ctx, hn, W("outputlayer.0.weight"));
    hn = ggml_add(ctx, hn, W("outputlayer.0.bias"));
    out.logits = ggml_mul_mat(ctx, W("outputlayer.1.weight"), hn);  // [outputdim,1]
    out.logits = ggml_add(ctx, out.logits, W("outputlayer.1.bias"));
    out.probs = ggml_sigmoid(ctx, out.logits);
    return out;
}

bool Ced::forward_from_tokens(const std::vector<float>& tokens, int n_tokens,
                              std::vector<float>& enc_norm,
                              std::vector<float>& logits,
                              std::vector<float>& probs, int n_threads) {
    const int C = (int)loader_.config().embed_dim;
    if ((int)tokens.size() != n_tokens * C) {
        std::fprintf(stderr, "ced: tokens size %zu != %d*%d\n", tokens.size(), n_tokens, C);
        return false;
    }
    BuildFn build = [&](ggml_context* ctx,
                        std::vector<GraphInput>& inputs) -> std::vector<ggml_tensor*> {
        // input tokens leaf: ne = [C, N]
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, n_tokens);
        ggml_set_input(x);
        inputs.push_back({x, tokens.data(), tokens.size() * sizeof(float)});
        Head h = build_blocks(ctx, x, n_tokens);
        return {h.enc, h.logits, h.probs};
    };
    std::vector<std::vector<float>> outs;
    if (!backend_->compute(n_threads, build, outs)) return false;
    enc_norm = std::move(outs[0]);
    logits = std::move(outs[1]);
    probs = std::move(outs[2]);
    return true;
}

bool Ced::mel_frontend(const std::vector<float>& wav, std::vector<float>& input_values,
                       int& T) {
    const float* win = loader_.host_f32("ced.mel_window");
    const float* fb = loader_.host_f32("ced.mel_filterbank");
    if (!win || !fb) {
        std::fprintf(stderr, "ced: missing baked mel window/filterbank\n");
        return false;
    }
    mel_spectrogram(loader_.config(), win, fb, wav, input_values, T);
    return true;
}

bool Ced::embed_from_input_values(const std::vector<float>& input_values, int T,
                                  std::vector<float>& init_bn_out,
                                  std::vector<float>& patch_embed, std::vector<float>& pos_out,
                                  std::vector<float>& tokens, int& n_tokens, int n_threads) {
    const CedConfig& c = loader_.config();
    if ((int)input_values.size() != (int)c.n_mels * T) {
        std::fprintf(stderr, "ced: input_values size %zu != %d*%d\n", input_values.size(),
                     (int)c.n_mels, T);
        return false;
    }
    BuildFn build = [&](ggml_context* ctx,
                        std::vector<GraphInput>& inputs) -> std::vector<ggml_tensor*> {
        Embed e = build_embed(ctx, inputs, input_values, T);
        return {e.bn, e.conv, e.pos, e.tokens};
    };
    std::vector<std::vector<float>> outs;
    if (!backend_->compute(n_threads, build, outs)) return false;
    init_bn_out = std::move(outs[0]);
    patch_embed = std::move(outs[1]);
    pos_out = std::move(outs[2]);
    tokens = std::move(outs[3]);
    n_tokens = (int)tokens.size() / (int)c.embed_dim;
    return true;
}

bool Ced::classify(const std::vector<float>& wav, std::vector<float>& logits,
                   std::vector<float>& probs, int n_threads) {
    const CedConfig& c = loader_.config();
    const int n_mels = (int)c.n_mels;
    const int chunk = (int)c.target_length;  // positional embeds cover this many frames
    std::vector<float> input_values;
    int T = 0;
    if (!mel_frontend(wav, input_values, T)) return false;

    // Clips longer than target_length are split into independent chunks (the
    // reference does this; time_pos_embed only spans target_length/patch_stride
    // positions). Per-chunk logits/probs are averaged into a single clip-level
    // result. T <= chunk is the single-chunk path, identical to the reference.
    const int n_chunks = (T <= chunk) ? 1 : (T + chunk - 1) / chunk;
    std::vector<double> logit_acc(c.outputdim, 0.0), prob_acc(c.outputdim, 0.0);
    int used = 0;
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int t0 = ci * chunk;
        const int clen = (T - t0) < chunk ? (T - t0) : chunk;
        if (clen < (int)c.patch_size) continue;  // skip sub-patch tail
        std::vector<float> civ((size_t)n_mels * clen);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < clen; ++t)
                civ[(size_t)m * clen + t] = input_values[(size_t)m * T + t0 + t];

        BuildFn build = [&](ggml_context* ctx,
                            std::vector<GraphInput>& inputs) -> std::vector<ggml_tensor*> {
            Embed e = build_embed(ctx, inputs, civ, clen);
            Head h = build_blocks(ctx, e.tokens, (int)e.tokens->ne[1]);
            return {h.logits, h.probs};
        };
        std::vector<std::vector<float>> outs;
        if (!backend_->compute(n_threads, build, outs)) return false;
        for (int i = 0; i < (int)c.outputdim; ++i) {
            logit_acc[i] += outs[0][i];
            prob_acc[i] += outs[1][i];
        }
        ++used;
    }
    if (used == 0) return false;
    logits.resize(c.outputdim);
    probs.resize(c.outputdim);
    for (int i = 0; i < (int)c.outputdim; ++i) {
        logits[i] = (float)(logit_acc[i] / used);
        probs[i] = (float)(prob_acc[i] / used);
    }
    return true;
}

}  // namespace ced
