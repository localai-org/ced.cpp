#include "ced_capi.h"

#include "audio_io.hpp"
#include "ced.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct CedContext {
    ced::Ced model;
    std::string last_error;
};

// Thread-local error for load-time failures (no context to attach to yet).
thread_local std::string g_load_error;

char* dup_cstr(const std::string& s) {
    char* p = (char*)std::malloc(s.size() + 1);
    if (p) std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

void json_escape(const std::string& in, std::string& out) {
    for (char ch : in) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default: out += ch;
        }
    }
}

std::vector<int> topk_indices(const std::vector<float>& v, int k) {
    const int n = (int)v.size();
    if (k <= 0 || k > n) k = n;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

// Run classify, resampling if needed. Returns false + sets ctx->last_error.
bool do_classify(CedContext* c, const float* samples, int n_samples, int sample_rate,
                 std::vector<float>& probs) {
    const int model_sr = (int)c->model.config().sample_rate;
    std::vector<float> wav(samples, samples + (n_samples > 0 ? n_samples : 0));
    if (sample_rate > 0 && sample_rate != model_sr)
        wav = ced::resample_linear(wav, sample_rate, model_sr);
    std::vector<float> logits;
    if (!c->model.classify(wav, logits, probs)) {
        c->last_error = "classify failed (audio too short or graph error)";
        return false;
    }
    c->last_error.clear();
    return true;
}

std::string tags_json(CedContext* c, const std::vector<float>& probs, int top_k) {
    const auto& labels = c->model.config().labels;
    std::vector<int> idx = topk_indices(probs, top_k);
    std::string out = "[";
    for (size_t i = 0; i < idx.size(); ++i) {
        int id = idx[i];
        std::string esc;
        if (id < (int)labels.size()) json_escape(labels[id], esc);
        char num[64];
        std::snprintf(num, sizeof(num), "{\"index\":%d,\"score\":%.6f,\"label\":\"", id,
                      probs[id]);
        out += num;
        out += esc;
        out += "\"}";
        if (i + 1 < idx.size()) out += ",";
    }
    out += "]";
    return out;
}

}  // namespace

extern "C" {

int ced_capi_abi_version(void) { return 1; }

ced_ctx* ced_capi_load(const char* gguf_path) {
    if (!gguf_path) {
        g_load_error = "null model path";
        return nullptr;
    }
    auto* c = new (std::nothrow) CedContext();
    if (!c) {
        g_load_error = "out of memory";
        return nullptr;
    }
    if (!c->model.load(gguf_path)) {
        g_load_error = std::string("failed to load model: ") + gguf_path;
        delete c;
        return nullptr;
    }
    g_load_error.clear();
    return reinterpret_cast<ced_ctx*>(c);
}

void ced_capi_free(ced_ctx* ctx) { delete reinterpret_cast<CedContext*>(ctx); }

const char* ced_capi_last_error(const ced_ctx* ctx) {
    if (!ctx) return g_load_error.c_str();
    return reinterpret_cast<const CedContext*>(ctx)->last_error.c_str();
}

int ced_capi_num_classes(const ced_ctx* ctx) {
    if (!ctx) return 0;
    return (int)reinterpret_cast<const CedContext*>(ctx)->model.config().labels.size();
}

const char* ced_capi_label(const ced_ctx* ctx, int index) {
    if (!ctx) return nullptr;
    const auto& labels = reinterpret_cast<const CedContext*>(ctx)->model.config().labels;
    if (index < 0 || index >= (int)labels.size()) return nullptr;
    return labels[index].c_str();
}

int ced_capi_sample_rate(const ced_ctx* ctx) {
    if (!ctx) return 0;
    return (int)reinterpret_cast<const CedContext*>(ctx)->model.config().sample_rate;
}

char* ced_capi_classify_pcm_json(ced_ctx* ctx, const float* samples, int n_samples,
                                 int sample_rate, int top_k) {
    auto* c = reinterpret_cast<CedContext*>(ctx);
    if (!c || !samples) return nullptr;
    std::vector<float> probs;
    if (!do_classify(c, samples, n_samples, sample_rate, probs)) return nullptr;
    return dup_cstr(tags_json(c, probs, top_k));
}

char* ced_capi_classify_path_json(ced_ctx* ctx, const char* wav_path, int top_k) {
    auto* c = reinterpret_cast<CedContext*>(ctx);
    if (!c || !wav_path) return nullptr;
    ced::Audio a;
    if (!ced::load_audio_mono(wav_path, a, (int)c->model.config().sample_rate)) {
        c->last_error = std::string("failed to read wav: ") + wav_path;
        return nullptr;
    }
    std::vector<float> probs;
    if (!do_classify(c, a.samples.data(), (int)a.samples.size(), a.sample_rate, probs))
        return nullptr;
    return dup_cstr(tags_json(c, probs, top_k));
}

int ced_capi_classify_pcm(ced_ctx* ctx, const float* samples, int n_samples, int sample_rate,
                          ced_tag* out, int max_tags) {
    auto* c = reinterpret_cast<CedContext*>(ctx);
    if (!c || !samples || !out || max_tags <= 0) return -1;
    std::vector<float> probs;
    if (!do_classify(c, samples, n_samples, sample_rate, probs)) return -1;
    const auto& labels = c->model.config().labels;
    std::vector<int> idx = topk_indices(probs, max_tags);
    int n = (int)idx.size();
    for (int i = 0; i < n; ++i) {
        out[i].index = idx[i];
        out[i].score = probs[idx[i]];
        out[i].label = idx[i] < (int)labels.size() ? labels[idx[i]].c_str() : "";
    }
    return n;
}

int ced_capi_classify_pcm_probs(ced_ctx* ctx, const float* samples, int n_samples,
                                int sample_rate, float* out, int n_out) {
    auto* c = reinterpret_cast<CedContext*>(ctx);
    if (!c || !samples || !out || n_out <= 0) return -1;
    std::vector<float> probs;
    if (!do_classify(c, samples, n_samples, sample_rate, probs)) return -1;
    const int n = std::min(n_out, (int)probs.size());
    std::memcpy(out, probs.data(), (size_t)n * sizeof(float));
    return n;
}

void ced_capi_free_string(char* s) { std::free(s); }

}  // extern "C"
