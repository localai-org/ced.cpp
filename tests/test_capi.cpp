// Smoke + parity test for the flat C-API: load -> classify PCM -> top tag.
// argv[1] = model gguf, argv[2] = baseline gguf
#include "ced_capi.h"
#include "parity.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string model = argc > 1 ? argv[1] : "models/ced-base-f32.gguf";
    std::string baseline = argc > 2 ? argv[2] : "tests/fixtures/ced-base.baseline.gguf";

    ced_ctx* ctx = ced_capi_load(model.c_str());
    if (!ctx) {
        std::fprintf(stderr, "FAIL: load: %s\n", ced_capi_last_error(nullptr));
        return 1;
    }
    bool ok = true;
    ok &= (ced_capi_abi_version() == 1);
    ok &= (ced_capi_num_classes(ctx) == 527);
    ok &= (ced_capi_sample_rate(ctx) == 16000);

    std::vector<float> wav, refp;
    std::vector<int64_t> shp;
    if (!cedtest::load_baseline(baseline, "audio_waveform", wav, shp) ||
        !cedtest::load_baseline(baseline, "probs", refp, shp)) {
        ced_capi_free(ctx);
        return 1;
    }
    int ref_argmax = 0;
    for (int i = 1; i < (int)refp.size(); ++i)
        if (refp[i] > refp[ref_argmax]) ref_argmax = i;

    // struct-array path
    ced_tag tags[5];
    int n = ced_capi_classify_pcm(ctx, wav.data(), (int)wav.size(), 16000, tags, 5);
    if (n <= 0) {
        std::fprintf(stderr, "FAIL: classify_pcm: %s\n", ced_capi_last_error(ctx));
        ced_capi_free(ctx);
        return 1;
    }
    std::fprintf(stderr, "top tag: [%d] %s score=%.4f (ref argmax=%d)\n", tags[0].index,
                 tags[0].label, tags[0].score, ref_argmax);
    ok &= (tags[0].index == ref_argmax);

    // JSON path
    char* json = ced_capi_classify_pcm_json(ctx, wav.data(), (int)wav.size(), 16000, 3);
    if (!json) { ok = false; } else {
        std::fprintf(stderr, "json: %s\n", json);
        ok &= (std::strstr(json, "\"label\"") != nullptr);
        ced_capi_free_string(json);
    }

    // all-scores path: same numbers as the sorted top-k path, in index order
    {
        const int nc = ced_capi_num_classes(ctx);
        std::vector<float> all(nc, -1.0f);
        int w = ced_capi_classify_pcm_probs(ctx, wav.data(), (int)wav.size(), 16000,
                                            all.data(), nc);
        ok &= (w == nc);
        std::vector<ced_tag> sorted(nc);
        int ns = ced_capi_classify_pcm(ctx, wav.data(), (int)wav.size(), 16000,
                                       sorted.data(), nc);
        ok &= (ns == nc);
        for (int i = 0; i < ns; ++i) ok &= (all[sorted[i].index] == sorted[i].score);
        // short buffer: only the first n_out classes
        float three[3];
        ok &= (ced_capi_classify_pcm_probs(ctx, wav.data(), (int)wav.size(), 16000,
                                           three, 3) == 3);
        ok &= (three[0] == all[0] && three[2] == all[2]);
        // errors
        ok &= (ced_capi_classify_pcm_probs(ctx, nullptr, 10, 16000, three, 3) == -1);
        ok &= (ced_capi_classify_pcm_probs(ctx, wav.data(), (int)wav.size(), 16000,
                                           nullptr, 3) == -1);
        std::fprintf(stderr, "classify_pcm_probs: %s\n", ok ? "ok" : "FAIL");
    }

    ced_capi_free(ctx);
    std::fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
