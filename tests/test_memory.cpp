// Tests for loading a model from memory (ced_capi_load_from_memory and the
// prefixed variant). No baseline fixture is needed: the reference is the same
// model loaded from its path, and outputs must match bit for bit.
//
// argv[1] = model gguf (any CED size), argv[2] = a second, different model gguf
//           (used as the other component of a bundle).
#include "ced_capi.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                                \
            std::fprintf(stderr, "\n");                                       \
            ++g_fail;                                                         \
        }                                                                     \
    } while (0)

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Deterministic 3 s test clip: a few tones plus a noise floor.
std::vector<float> make_wav() {
    std::vector<float> w(16000 * 3);
    uint32_t s = 12345;
    for (size_t i = 0; i < w.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        float noise = ((s >> 8) / (float)(1 << 24) - 0.5f) * 0.05f;
        float t = (float)i / 16000.0f;
        w[i] = 0.3f * std::sin(6.2831853f * 440.0f * t) + 0.2f * std::sin(6.2831853f * 1250.0f * t * (1 + t)) + noise;
    }
    return w;
}

bool scores(ced_ctx* c, const std::vector<float>& wav, std::vector<float>& out) {
    int n = ced_capi_num_classes(c);
    out.assign(n, 0.0f);
    return ced_capi_classify_pcm_probs(c, wav.data(), (int)wav.size(), 16000, out.data(), n) == n;
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Copy `n` bytes of `src` into a buffer of exactly that size, so ASan flags any
// read past the end.
std::vector<uint8_t> exact_copy(const std::vector<uint8_t>& src, size_t n) {
    return std::vector<uint8_t>(src.begin(), src.begin() + n);
}

// ---- in-memory bundle builder ------------------------------------------------

void copy_kv(gguf_context* out, gguf_context* g, int64_t i, const std::string& name) {
    const char* k = name.c_str();
    switch (gguf_get_kv_type(g, i)) {
        case GGUF_TYPE_UINT32:  gguf_set_val_u32(out, k, gguf_get_val_u32(g, i)); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32(out, k, gguf_get_val_f32(g, i)); break;
        case GGUF_TYPE_BOOL:    gguf_set_val_bool(out, k, gguf_get_val_bool(g, i)); break;
        case GGUF_TYPE_STRING:  gguf_set_val_str(out, k, gguf_get_val_str(g, i)); break;
        case GGUF_TYPE_ARRAY: {
            const gguf_type at = gguf_get_arr_type(g, i);
            const size_t n = gguf_get_arr_n(g, i);
            if (at == GGUF_TYPE_STRING) {
                std::vector<const char*> v(n);
                for (size_t j = 0; j < n; ++j) v[j] = gguf_get_arr_str(g, i, j);
                gguf_set_arr_str(out, k, v.data(), n);
            } else {
                gguf_set_arr_data(out, k, at, gguf_get_arr_data(g, i), n);
            }
            break;
        }
        default: break;
    }
}

struct Component {
    std::string prefix;
    const std::vector<uint8_t>* gguf;
    std::string retype_key;  // if set, this key is written as a float instead
};

// Builds one GGUF holding every component, each under its prefix. Keys and
// tensors are renamed `<prefix><name>`; tensor bytes are copied unchanged.
std::vector<uint8_t> build_bundle(const std::vector<Component>& comps) {
    gguf_context* out = gguf_init_empty();
    gguf_set_val_str(out, "general.architecture", "test-bundle");
    std::vector<gguf_context*> srcs;
    std::vector<ggml_context*> ctxs;
    ggml_context* tctx = ggml_init({ggml_tensor_overhead() * 4096, nullptr, true});
    std::vector<const void*> datas;
    for (auto& c : comps) {
        ggml_context* sc = nullptr;
        gguf_context* g = gguf_init_from_buffer(c.gguf->data(), c.gguf->size(), {false, &sc});
        if (!g) { std::fprintf(stderr, "bundle: bad source\n"); std::exit(2); }
        srcs.push_back(g);
        ctxs.push_back(sc);
        for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
            std::string k = gguf_get_key(g, i);
            if (k == "general.alignment") continue;
            if (!c.retype_key.empty() && k == c.retype_key) gguf_set_val_f32(out, (c.prefix + k).c_str(), 1.0f);
            else copy_kv(out, g, i, c.prefix + k);
        }
        for (ggml_tensor* t = ggml_get_first_tensor(sc); t; t = ggml_get_next_tensor(sc, t)) {
            ggml_tensor* d = ggml_new_tensor(tctx, t->type, GGML_MAX_DIMS, t->ne);
            ggml_set_name(d, (c.prefix + ggml_get_name(t)).c_str());
            d->data = t->data;
            gguf_add_tensor(out, d);
            datas.push_back(t->data);
        }
    }
    const size_t meta = gguf_get_meta_size(out);
    std::vector<uint8_t> buf(meta, 0);
    gguf_get_meta_data(out, buf.data());
    const size_t align = 32;
    for (int64_t i = 0; i < gguf_get_n_tensors(out); ++i) {
        const size_t off = meta + gguf_get_tensor_offset(out, i);
        if (buf.size() < off) buf.resize(off, 0);
        const uint8_t* p = (const uint8_t*)datas[i];
        buf.insert(buf.end(), p, p + gguf_get_tensor_size(out, i));
        buf.resize((buf.size() + align - 1) / align * align, 0);
    }
    for (auto g : srcs) gguf_free(g);
    for (auto c : ctxs) ggml_free(c);
    ggml_free(tctx);
    gguf_free(out);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf other-model.gguf\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1], path2 = argv[2];
    const std::vector<uint8_t> file = read_file(path), file2 = read_file(path2);
    CHECK(!file.empty() && !file2.empty(), "cannot read models");
    const std::vector<float> wav = make_wav();

    // Reference: path loader.
    ced_ctx* ref = ced_capi_load(path.c_str());
    ced_ctx* ref2 = ced_capi_load(path2.c_str());
    if (!ref || !ref2) { std::fprintf(stderr, "FAIL: path load\n"); return 1; }
    std::vector<float> ref_s, ref_s2;
    CHECK(scores(ref, wav, ref_s) && scores(ref2, wav, ref_s2), "reference classify");
    CHECK(!bitwise_equal(ref_s, ref_s2), "the two test models must differ");

    // 1. Same file from memory: identical scores, labels and rate.
    {
        std::vector<uint8_t> heap = file;
        ced_ctx* m = ced_capi_load_from_memory(heap.data(), heap.size());
        CHECK(m != nullptr, "load_from_memory: %s", ced_capi_last_error(nullptr));
        if (m) {
            // 2. The buffer is not used after the call: wipe and free it first.
            std::memset(heap.data(), 0xA5, heap.size());
            std::vector<uint8_t>().swap(heap);
            std::vector<float> s;
            CHECK(scores(m, wav, s) && bitwise_equal(s, ref_s), "memory scores differ from path scores");
            CHECK(ced_capi_num_classes(m) == ced_capi_num_classes(ref), "num_classes");
            CHECK(ced_capi_sample_rate(m) == ced_capi_sample_rate(ref), "sample_rate");
            CHECK(std::strcmp(ced_capi_label(m, 0), ced_capi_label(ref, 0)) == 0, "label 0");
            ced_capi_free(m);
        }
    }

    // 3. Prefixed view of a two-component bundle, both orders.
    for (int order = 0; order < 2; ++order) {
        std::vector<Component> comps = order == 0
            ? std::vector<Component>{{"ced.", &file, ""}, {"other.", &file2, ""}}
            : std::vector<Component>{{"other.", &file2, ""}, {"ced.", &file, ""}};
        std::vector<uint8_t> bundle = build_bundle(comps);
        ced_ctx* a = ced_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "ced.");
        ced_ctx* b = ced_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "other.");
        CHECK(a && b, "prefixed load (order %d): %s", order, ced_capi_last_error(nullptr));
        std::memset(bundle.data(), 0x5A, bundle.size());
        std::vector<uint8_t>().swap(bundle);
        if (a && b) {
            std::vector<float> sa, sb;
            CHECK(scores(a, wav, sa) && bitwise_equal(sa, ref_s), "prefixed 'ced.' scores (order %d)", order);
            CHECK(scores(b, wav, sb) && bitwise_equal(sb, ref_s2), "prefixed 'other.' scores (order %d)", order);
        }
        ced_capi_free(a);
        ced_capi_free(b);
    }

    // 4. Bad arguments and the wrong prefix.
    CHECK(ced_capi_load_from_memory(nullptr, 100) == nullptr, "NULL data");
    CHECK(std::strlen(ced_capi_last_error(nullptr)) > 0, "error set for NULL data");
    CHECK(ced_capi_load_from_memory(file.data(), 0) == nullptr, "size 0");
    CHECK(ced_capi_load_from_memory_prefixed(file.data(), file.size(), nullptr) == nullptr, "NULL prefix");
    CHECK(ced_capi_load_from_memory_prefixed(file.data(), file.size(), "") == nullptr, "empty prefix");
    {
        std::vector<uint8_t> bundle = build_bundle({{"ced.", &file, ""}});
        CHECK(ced_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "nope.") == nullptr, "wrong prefix");
        CHECK(std::strstr(ced_capi_last_error(nullptr), "no tensors under prefix") != nullptr, "wrong prefix message: %s",
              ced_capi_last_error(nullptr));
        // A standalone file has no prefix at all.
        CHECK(ced_capi_load_from_memory_prefixed(file.data(), file.size(), "ced.") == nullptr, "prefix on standalone file");
        // Not a bundle: loading it without a prefix finds no model keys.
        CHECK(ced_capi_load_from_memory(bundle.data(), bundle.size()) == nullptr, "bundle without prefix");
        CHECK(std::strstr(ced_capi_last_error(nullptr), "not a ced model") != nullptr, "bundle w/o prefix message: %s",
              ced_capi_last_error(nullptr));
    }

    // 5. A key with the wrong type is an error, not an abort.
    {
        std::vector<uint8_t> bad = build_bundle({{"ced.", &file, "ced.depth"}});
        CHECK(ced_capi_load_from_memory_prefixed(bad.data(), bad.size(), "ced.") == nullptr, "retyped key (prefixed)");
        CHECK(std::strstr(ced_capi_last_error(nullptr), "unexpected type") != nullptr, "message: %s", ced_capi_last_error(nullptr));
        std::vector<uint8_t> bad2 = build_bundle({{"", &file, "ced.depth"}});
        CHECK(ced_capi_load_from_memory(bad2.data(), bad2.size()) == nullptr, "retyped key (plain)");
    }

    // 6. Truncation at every region: header (dense), tensor table, tensor data.
    {
        std::vector<uint8_t> bundle = build_bundle({{"ced.", &file, ""}, {"other.", &file2, ""}});
        struct Case { const char* name; const std::vector<uint8_t>* buf; const char* prefix; };
        const Case cases[] = {{"plain", &file, nullptr}, {"bundle", &bundle, "ced."}};
        for (const Case& cs : cases) {
            const size_t total = cs.buf->size();
            std::vector<size_t> cuts;
            for (size_t n = 0; n < 512 && n < total; ++n) cuts.push_back(n);
            for (size_t n = 512; n < 200000 && n < total; n += 257) cuts.push_back(n);
            for (int i = 1; i < 120; ++i) cuts.push_back(total * i / 120);
            cuts.push_back(total - 1);
            int accepted = 0;
            for (size_t n : cuts) {
                std::vector<uint8_t> t = exact_copy(*cs.buf, n);
                ced_ctx* c = cs.prefix ? ced_capi_load_from_memory_prefixed(t.data(), t.size(), cs.prefix)
                                       : ced_capi_load_from_memory(t.data(), t.size());
                if (c) {
                    ++accepted;
                    ced_capi_free(c);
                } else {
                    CHECK(std::strlen(ced_capi_last_error(nullptr)) > 0, "%s: no error text at %zu", cs.name, n);
                }
            }
            // A plain file cut anywhere must be rejected. A bundle view may succeed
            // only once the cut is past the bytes of the selected component.
            if (!cs.prefix) CHECK(accepted == 0, "%s: %d truncated buffers accepted", cs.name, accepted);
            else std::fprintf(stderr, "[%s] %zu cuts, %d accepted (cut after the component)\n", cs.name, cuts.size(), accepted);
        }
        // Cut inside the selected component (first in the bundle): must be rejected.
        std::vector<uint8_t> first = build_bundle({{"ced.", &file, ""}, {"other.", &file2, ""}});
        std::vector<uint8_t> t = exact_copy(first, first.size() / 10 + 100000 < first.size() ? first.size() / 10 + 100000 : first.size() / 2);
        CHECK(ced_capi_load_from_memory_prefixed(t.data(), t.size(), "ced.") == nullptr, "cut inside component");
        CHECK(std::strstr(ced_capi_last_error(nullptr), "truncated") != nullptr ||
              std::strstr(ced_capi_last_error(nullptr), "corrupt") != nullptr, "cut message: %s", ced_capi_last_error(nullptr));
    }

    // 7. Corrupt headers.
    {
        auto expect_reject = [&](const char* what, std::vector<uint8_t> b) {
            std::vector<uint8_t> t = exact_copy(b, b.size());
            CHECK(ced_capi_load_from_memory(t.data(), t.size()) == nullptr, "%s accepted", what);
            CHECK(std::strlen(ced_capi_last_error(nullptr)) > 0, "%s: no error text", what);
        };
        std::vector<uint8_t> b = file;
        b[0] ^= 0xff;                       expect_reject("bad magic", b);
        b = file; b[4] = 0xff; b[5] = 0xff; expect_reject("bad version", b);
        b = file; std::memset(&b[8], 0xff, 8);  expect_reject("huge tensor count", b);
        b = file; std::memset(&b[16], 0xff, 8); expect_reject("huge kv count", b);
        b = file; std::memset(&b[8], 0, 8);     expect_reject("zero tensors", b);
        b.assign(file.size(), 0);               expect_reject("all zero", b);
        b.assign(file.size(), 0xff);            expect_reject("all 0xff", b);
        b.assign(8, 0);                         expect_reject("8 zero bytes", b);
    }

    // 8. Bit flips in the header and tensor table must never crash.
    {
        uint32_t s = 777;
        auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
        int ok = 0, rejected = 0;
        const size_t span = std::min<size_t>(file.size(), 4096);
        for (int it = 0; it < 400; ++it) {
            std::vector<uint8_t> b = exact_copy(file, file.size());
            for (int k = 0; k < 1 + (int)(rnd() % 4); ++k) b[rnd() % span] ^= (uint8_t)(1u << (rnd() % 8));
            ced_ctx* c = ced_capi_load_from_memory(b.data(), b.size());
            if (c) { ++ok; ced_capi_free(c); } else ++rejected;
        }
        std::fprintf(stderr, "[bitflip] %d accepted, %d rejected, no crash\n", ok, rejected);
    }

    // 9. Concurrent loads from memory (plain and prefixed) match the reference.
    {
        std::vector<uint8_t> bundle = build_bundle({{"ced.", &file, ""}, {"other.", &file2, ""}});
        std::atomic<int> bad{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 6; ++t)
            th.emplace_back([&, t]() {
                for (int i = 0; i < 3; ++i) {
                    ced_ctx* c;
                    const std::vector<float>* want;
                    if (t % 3 == 0) { c = ced_capi_load_from_memory(file.data(), file.size()); want = &ref_s; }
                    else if (t % 3 == 1) { c = ced_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "ced."); want = &ref_s; }
                    else { c = ced_capi_load_from_memory_prefixed(bundle.data(), bundle.size(), "other."); want = &ref_s2; }
                    std::vector<float> s;
                    if (!c || !scores(c, wav, s) || !bitwise_equal(s, *want)) ++bad;
                    ced_capi_free(c);
                    // Each thread has its own load error.
                    if (ced_capi_load_from_memory(nullptr, 0) != nullptr || std::strlen(ced_capi_last_error(nullptr)) == 0) ++bad;
                }
            });
        for (auto& x : th) x.join();
        CHECK(bad == 0, "%d concurrent failures", bad.load());
    }

    ced_capi_free(ref);
    ced_capi_free(ref2);
    std::fprintf(stderr, "%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
