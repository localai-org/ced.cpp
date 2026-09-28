#pragma once
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cedtest {

// Load an f32 tensor (flattened, row-major) by name from a baseline gguf.
inline bool load_baseline(const std::string& path, const std::string& name,
                          std::vector<float>& out, std::vector<int64_t>& shape) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        std::fprintf(stderr, "[parity] tensor '%s' not found in %s\n", name.c_str(), path.c_str());
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    shape.clear();
    for (int i = ggml_n_dims(t) - 1; i >= 0; --i) shape.push_back(t->ne[i]);
    size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    std::memcpy(out.data(), t->data, n * sizeof(float));
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

// Compare got vs ref; prints stats; returns true if all within tolerance.
inline bool compare(const std::vector<float>& got, const std::vector<float>& ref,
                    const char* label, float atol, float rtol) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu\n", label, got.size(), ref.size());
        return false;
    }
    double maxabs = 0.0, sumabs = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double)got[i] - (double)ref[i]);
        sumabs += d;
        if (d > maxabs) { maxabs = d; worst = i; }
    }
    double mean = sumabs / (got.size() ? got.size() : 1);
    bool ok = true;
    for (size_t i = 0; i < got.size() && ok; ++i) {
        double tol = (double)atol + (double)rtol * std::fabs((double)ref[i]);
        if (std::fabs((double)got[i] - (double)ref[i]) > tol) ok = false;
    }
    std::fprintf(stderr, "[%s] n=%zu max|d|=%.3e mean|d|=%.3e (worst@%zu got=%.5f ref=%.5f) -> %s\n",
                 label, got.size(), maxabs, mean, worst, got[worst], ref[worst], ok ? "OK" : "FAIL");
    return ok;
}

// Tolerance for an intermediate stage (activations of magnitude ~1-10). The
// reference values are CPU f32 and are gated tightly on CPU. GPU backends run
// their matmuls at lower internal precision (ggml's Metal matmul stages tiles
// in half), which moves intermediates by up to ~1e-2 while the final
// probabilities still agree to ~1e-4, so off-CPU the stage gate is widened.
// The end-to-end probability gates are the same on every device.
inline float stage_tol(const std::string& device, float cpu_tol) {
    const float gpu_tol = 2e-2f;
    return device == "cpu" ? cpu_tol : (cpu_tol > gpu_tol ? cpu_tol : gpu_tol);
}

}  // namespace cedtest
