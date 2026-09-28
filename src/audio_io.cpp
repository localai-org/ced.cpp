#ifndef CED_EXTERNAL_DR_WAV
#define DR_WAV_IMPLEMENTATION  // an embedding project may provide it instead
#endif
#include "dr_wav.h"

#include "audio_io.hpp"

#include <cmath>

namespace ced {

std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr) {
    if (in_sr == out_sr || in.empty()) return in;
    const double ratio = (double)out_sr / (double)in_sr;
    const size_t n_out = (size_t)std::floor(in.size() * ratio);
    std::vector<float> out(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        const double src = i / ratio;
        const size_t i0 = (size_t)src;
        const double frac = src - i0;
        const float a = in[i0];
        const float b = (i0 + 1 < in.size()) ? in[i0 + 1] : a;
        out[i] = (float)(a + (b - a) * frac);
    }
    return out;
}

bool load_audio_mono(const std::string& path, Audio& out, int target_sr) {
    unsigned int ch = 0, sr = 0;
    drwav_uint64 frames = 0;
    float* pcm = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &ch, &sr, &frames, nullptr);
    if (!pcm) return false;
    std::vector<float> mono(frames);
    for (drwav_uint64 i = 0; i < frames; ++i) {
        double acc = 0;
        for (unsigned int c = 0; c < ch; ++c) acc += pcm[i * ch + c];
        mono[i] = (float)(acc / (ch ? ch : 1));
    }
    drwav_free(pcm, nullptr);
    out.samples = resample_linear(mono, (int)sr, target_sr);
    out.sample_rate = target_sr;
    return true;
}

}  // namespace ced
