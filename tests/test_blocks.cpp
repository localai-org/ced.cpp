// Parity gate: the 12 ViT blocks + final norm + mean-pool head.
//
// Feeds the baseline `tokens_in` (post patch-embed + positional) through the C++
// ggml graph and asserts enc_norm / logits / probs match the PyTorch reference
// dump. This gates the hardest numerics (GELU-erf, LayerNorm eps, attention).
//
// argv[1] = model gguf (default models/ced-base-f32.gguf)
// argv[2] = baseline gguf (default tests/fixtures/ced-base.baseline.gguf)
#include "ced.hpp"
#include "parity.hpp"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string model = argc > 1 ? argv[1] : "models/ced-base-f32.gguf";
    std::string baseline = argc > 2 ? argv[2] : "tests/fixtures/ced-base.baseline.gguf";

    ced::Ced m;
    if (!m.load(model)) {
        std::fprintf(stderr, "FAIL: could not load model %s\n", model.c_str());
        return 1;
    }
    std::fprintf(stderr, "device: %s\n", m.device_name().c_str());

    std::vector<float> tokens;
    std::vector<int64_t> shape;
    if (!cedtest::load_baseline(baseline, "tokens_in", tokens, shape)) return 1;
    // tokens_in shape is [N, C]
    if (shape.size() != 2) {
        std::fprintf(stderr, "FAIL: tokens_in is not 2D\n");
        return 1;
    }
    int N = (int)shape[0];

    std::vector<float> enc, logits, probs;
    if (!m.forward_from_tokens(tokens, N, enc, logits, probs)) {
        std::fprintf(stderr, "FAIL: forward_from_tokens failed\n");
        return 1;
    }

    bool ok = true;
    std::vector<float> ref;
    std::vector<int64_t> rshape;

    if (cedtest::load_baseline(baseline, "enc_norm", ref, rshape))
        ok &= cedtest::compare(enc, ref, "enc_norm", cedtest::stage_tol(m.device_name(), 2e-3f), 0.0f);
    else ok = false;

    if (cedtest::load_baseline(baseline, "logits", ref, rshape))
        ok &= cedtest::compare(logits, ref, "logits", cedtest::stage_tol(m.device_name(), 2e-3f), 0.0f);
    else ok = false;

    if (cedtest::load_baseline(baseline, "probs", ref, rshape))
        ok &= cedtest::compare(probs, ref, "probs", 1e-3f, 0.0f);
    else ok = false;

    std::fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
