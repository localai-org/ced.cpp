// ced-cli: classify everyday sounds in an audio file using a CED GGUF model.
//
//   ced-cli info <model.gguf>
//   ced-cli classify <model.gguf> <audio.wav> [--top-k N] [--threads N]
//   ced-cli bench <model.gguf> <audio.wav> [--iters N] [--warmup W] [--threads N]
#include "audio_io.hpp"
#include "ced.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static int cmd_info(const char* path) {
    ced::Ced m;
    if (!m.load(path)) {
        std::fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }
    const ced::CedConfig& c = m.config();
    std::printf("ced.cpp model: %s\n", path);
    std::printf("  arch              : %s\n", c.arch.c_str());
    std::printf("  embed/depth/heads : %u / %u / %u\n", c.embed_dim, c.depth, c.num_heads);
    std::printf("  classes           : %u\n", c.outputdim);
    std::printf("  pooling           : %s\n", c.pooling.c_str());
    std::printf("  mel               : n_mels=%u n_fft=%u hop=%u win=%u sr=%u\n", c.n_mels,
                c.n_fft, c.hop_size, c.win_size, c.sample_rate);
    std::printf("  target_length     : %u frames\n", c.target_length);
    std::printf("  device            : %s\n", m.device_name().c_str());
    return 0;
}

static int cmd_classify(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: ced-cli classify <model.gguf> <audio.wav> [--top-k N] [--threads N]\n");
        return 1;
    }
    const char* model = argv[2];
    const char* audio = argv[3];
    int top_k = 10, threads = 4;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    }

    ced::Ced m;
    if (!m.load(model)) {
        std::fprintf(stderr, "failed to load %s\n", model);
        return 1;
    }
    ced::Audio a;
    if (!ced::load_audio_mono(audio, a, (int)m.config().sample_rate)) {
        std::fprintf(stderr, "failed to read audio %s\n", audio);
        return 1;
    }
    std::vector<float> logits, probs;
    if (!m.classify(a.samples, logits, probs, threads)) {
        std::fprintf(stderr, "classify failed\n");
        return 1;
    }

    std::vector<int> idx(probs.size());
    std::iota(idx.begin(), idx.end(), 0);
    int k = std::min<int>(top_k, (int)probs.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int x, int y) { return probs[x] > probs[y]; });
    const auto& labels = m.config().labels;
    std::printf("# %s (%.1fs)\n", audio, a.samples.size() / (float)a.sample_rate);
    for (int i = 0; i < k; ++i) {
        int id = idx[i];
        std::printf("%.4f  %s\n", probs[id], id < (int)labels.size() ? labels[id].c_str() : "?");
    }
    return 0;
}

static int cmd_bench(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: ced-cli bench <model.gguf> <audio.wav> [--iters N] [--warmup W] [--threads N]\n");
        return 1;
    }
    const char* model = argv[2];
    const char* audio = argv[3];
    int iters = 30, warmup = 5, threads = 4;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    }

    ced::Ced m;
    if (!m.load(model)) {
        std::fprintf(stderr, "failed to load %s\n", model);
        return 1;
    }
    ced::Audio a;
    if (!ced::load_audio_mono(audio, a, (int)m.config().sample_rate)) {
        std::fprintf(stderr, "failed to read audio %s\n", audio);
        return 1;
    }
    const double clip_s = a.samples.size() / (double)a.sample_rate;

    std::vector<float> logits, probs;
    for (int i = 0; i < warmup; ++i) m.classify(a.samples, logits, probs, threads);

    std::vector<double> ms;
    ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        if (!m.classify(a.samples, logits, probs, threads)) {
            std::fprintf(stderr, "classify failed\n");
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    double sum = 0;
    for (double v : ms) sum += v;
    double mean = sum / ms.size();
    double median = ms[ms.size() / 2];
    double minv = ms.front(), maxv = ms.back();

    std::printf("model=%s  device=%s  clip=%.2fs  threads=%d  iters=%d\n", model,
                m.device_name().c_str(), clip_s, threads, iters);
    std::printf("  latency ms: min=%.2f  median=%.2f  mean=%.2f  max=%.2f\n", minv, median, mean, maxv);
    std::printf("  RTF (clip_s/mean_s): %.1fx realtime  (%.1f clips/s)\n",
                clip_s / (mean / 1000.0), 1000.0 / mean);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ced-cli <info|classify|bench> ...\n");
        return 1;
    }
    if (!std::strcmp(argv[1], "info") && argc >= 3) return cmd_info(argv[2]);
    if (!std::strcmp(argv[1], "classify")) return cmd_classify(argc, argv);
    if (!std::strcmp(argv[1], "bench")) return cmd_bench(argc, argv);
    // bare form: ced-cli <model> <audio>
    if (argc >= 3) {
        char* shim[] = {argv[0], (char*)"classify", argv[1], argv[2]};
        return cmd_classify(4, shim);
    }
    std::fprintf(stderr, "usage: ced-cli <info|classify> ...\n");
    return 1;
}
