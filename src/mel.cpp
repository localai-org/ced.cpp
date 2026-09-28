#include "mel.hpp"

#include "fft.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ced {

// Reflect-pad (no edge repeat, torch 'reflect' semantics): mirror around the
// boundary samples. Requires pad < len.
static std::vector<float> reflect_pad(const std::vector<float>& x, int pad) {
    const int n = (int)x.size();
    std::vector<float> out(n + 2 * pad);
    for (int i = 0; i < pad; ++i) out[i] = x[pad - i];                 // left
    for (int i = 0; i < n; ++i) out[pad + i] = x[i];                   // center
    for (int i = 0; i < pad; ++i) out[pad + n + i] = x[n - 2 - i];     // right
    return out;
}

void mel_spectrogram(const CedConfig& cfg, const float* window, const float* filterbank,
                     const std::vector<float>& wav, std::vector<float>& input_values, int& T) {
    const int n_fft = (int)cfg.n_fft;
    const int hop = (int)cfg.hop_size;
    const int n_mels = (int)cfg.n_mels;
    const int n_freqs = (int)cfg.n_freqs;  // n_fft/2 + 1
    const int pad = n_fft / 2;             // center=True

    std::vector<float> padded = cfg.center ? reflect_pad(wav, pad) : wav;
    // torch.stft frame count for center=True: 1 + n_samples / hop
    T = 1 + (int)wav.size() / hop;

    input_values.assign((size_t)n_mels * T, 0.0f);
    std::vector<float> frame(n_fft), re, im, power(n_freqs);

    // Each triangular mel filter is non-zero over a narrow band only; restrict
    // the filterbank product to [lo, hi) per row. The skipped terms are exact
    // zeros, so the sums (and their order) are unchanged.
    std::vector<int> lo(n_mels, 0), hi(n_mels, 0);
    for (int m = 0; m < n_mels; ++m) {
        const float* fb = filterbank + (size_t)m * n_freqs;
        int f0 = 0, f1 = n_freqs;
        while (f0 < n_freqs && fb[f0] == 0.0f) ++f0;
        while (f1 > f0 && fb[f1 - 1] == 0.0f) --f1;
        lo[m] = f0;
        hi[m] = f1;
    }

    for (int t = 0; t < T; ++t) {
        const int start = t * hop;
        for (int i = 0; i < n_fft; ++i) {
            float s = (start + i < (int)padded.size()) ? padded[start + i] : 0.0f;
            frame[i] = s * window[i];
        }
        rfft(frame, re, im);  // bins [0, n_freqs)
        for (int f = 0; f < n_freqs; ++f) power[f] = re[f] * re[f] + im[f] * im[f];
        // mel = filterbank[n_mels, n_freqs] @ power
        for (int m = 0; m < n_mels; ++m) {
            const float* fb = filterbank + (size_t)m * n_freqs;
            float acc = 0.0f;
            for (int f = lo[m]; f < hi[m]; ++f) acc += fb[f] * power[f];
            input_values[(size_t)m * T + t] = acc;  // mel power, dB applied below
        }
    }

    // AmplitudeToDB(stype="power", top_db=120): db = mult*log10(max(x, amin));
    // ref power 1.0 -> no offset. Then clamp to (global max db - top_db).
    const float mult = cfg.a2db_multiplier;  // 10
    const float amin = cfg.a2db_amin;        // 1e-10
    float gmax = -INFINITY;
    for (size_t i = 0; i < input_values.size(); ++i) {
        float db = mult * std::log10(std::max(input_values[i], amin));
        input_values[i] = db;
        if (db > gmax) gmax = db;
    }
    const float floor_db = gmax - cfg.a2db_top_db;
    for (size_t i = 0; i < input_values.size(); ++i)
        input_values[i] = std::max(input_values[i], floor_db);
}

}  // namespace ced
