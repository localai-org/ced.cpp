// End-to-end quantization parity: run the full waveform->probs chain on a
// (possibly quantized) model and compare to the f32 PyTorch reference probs.
// A tagger only needs (a) probs close within tolerance and (b) the top-k
// predicted classes unchanged, so we check both.
//
// argv[1] = model gguf, argv[2] = baseline gguf, argv[3] = abs tolerance (probs)
#include "ced.hpp"
#include "parity.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

static std::vector<int> topk(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

int main(int argc, char** argv) {
    std::string model = argc > 1 ? argv[1] : "models/ced-base-q8_0.gguf";
    std::string baseline = argc > 2 ? argv[2] : "tests/fixtures/ced-base.baseline.gguf";
    float tol = argc > 3 ? (float)std::atof(argv[3]) : 3e-2f;

    ced::Ced m;
    if (!m.load(model)) { std::fprintf(stderr, "FAIL: load %s\n", model.c_str()); return 1; }
    std::fprintf(stderr, "device: %s\n", m.device_name().c_str());

    std::vector<float> wav, refp;
    std::vector<int64_t> shp;
    if (!cedtest::load_baseline(baseline, "audio_waveform", wav, shp)) return 1;
    if (!cedtest::load_baseline(baseline, "probs", refp, shp)) return 1;

    std::vector<float> logits, probs;
    if (!m.classify(wav, logits, probs)) { std::fprintf(stderr, "FAIL: classify\n"); return 1; }

    bool ok = cedtest::compare(probs, refp, "quant_probs", tol, 0.0f);

    const int K = 5;
    std::vector<int> tk_got = topk(probs, K), tk_ref = topk(refp, K);
    bool topk_match = (tk_got == tk_ref);
    std::fprintf(stderr, "top-%d indices %s  (ref:", K, topk_match ? "MATCH" : "DIFFER");
    for (int i : tk_ref) std::fprintf(stderr, " %d", i);
    std::fprintf(stderr, " | got:");
    for (int i : tk_got) std::fprintf(stderr, " %d", i);
    std::fprintf(stderr, ")\n");

    const auto& labels = m.config().labels;
    std::fprintf(stderr, "top-%d tags:\n", K);
    for (int i : tk_got)
        std::fprintf(stderr, "  %.4f  %s\n", probs[i],
                     i < (int)labels.size() ? labels[i].c_str() : "?");

    ok &= topk_match;
    std::fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
