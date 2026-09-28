// Parity gates for the front of the pipeline + end-to-end:
//   mel frontend  : audio_waveform -> input_values
//   embed         : input_values   -> init_bn_out, patch_embed, pos_out, tokens_in
//   end-to-end    : audio_waveform -> probs (full chain)
//
// argv[1] = model gguf, argv[2] = baseline gguf
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
        std::fprintf(stderr, "FAIL: load %s\n", model.c_str());
        return 1;
    }
    std::fprintf(stderr, "device: %s\n", m.device_name().c_str());

    std::vector<float> wav, ref;
    std::vector<int64_t> shp;
    if (!cedtest::load_baseline(baseline, "audio_waveform", wav, shp)) return 1;

    bool ok = true;

    // --- mel frontend ---
    std::vector<float> input_values;
    int T = 0;
    if (!m.mel_frontend(wav, input_values, T)) { std::fprintf(stderr, "FAIL: mel\n"); return 1; }
    if (cedtest::load_baseline(baseline, "input_values", ref, shp))
        ok &= cedtest::compare(input_values, ref, "input_values", 2e-3f, 0.0f);
    else ok = false;

    // --- embed (init_bn + patch_embed + pos + flatten) ---
    std::vector<float> init_bn_out, patch_embed, pos_out, tokens;
    int n_tokens = 0;
    if (!m.embed_from_input_values(input_values, T, init_bn_out, patch_embed, pos_out, tokens,
                                   n_tokens)) {
        std::fprintf(stderr, "FAIL: embed\n");
        return 1;
    }
    if (cedtest::load_baseline(baseline, "init_bn_out", ref, shp))
        ok &= cedtest::compare(init_bn_out, ref, "init_bn_out", cedtest::stage_tol(m.device_name(), 2e-3f), 0.0f);
    else ok = false;
    if (cedtest::load_baseline(baseline, "patch_embed", ref, shp))
        ok &= cedtest::compare(patch_embed, ref, "patch_embed", cedtest::stage_tol(m.device_name(), 3e-3f), 0.0f);
    else ok = false;
    if (cedtest::load_baseline(baseline, "pos_out", ref, shp))
        ok &= cedtest::compare(pos_out, ref, "pos_out", cedtest::stage_tol(m.device_name(), 3e-3f), 0.0f);
    else ok = false;
    if (cedtest::load_baseline(baseline, "tokens_in", ref, shp))
        ok &= cedtest::compare(tokens, ref, "tokens_in", cedtest::stage_tol(m.device_name(), 3e-3f), 0.0f);
    else ok = false;

    // --- end-to-end: waveform -> probs ---
    std::vector<float> logits, probs;
    if (!m.classify(wav, logits, probs)) { std::fprintf(stderr, "FAIL: classify\n"); return 1; }
    if (cedtest::load_baseline(baseline, "probs", ref, shp))
        ok &= cedtest::compare(probs, ref, "e2e_probs", 2e-3f, 0.0f);
    else ok = false;

    std::fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
