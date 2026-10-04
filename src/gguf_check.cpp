#include "gguf_check.hpp"

#include <cstdint>
#include <cstring>

namespace ced {

namespace {

struct Cursor {
    const uint8_t* p;
    size_t size;
    size_t pos = 0;

    bool have(uint64_t n) const { return n <= size - pos; }
    bool skip(uint64_t n) {
        if (!have(n)) return false;
        pos += (size_t)n;
        return true;
    }
    template <typename T>
    bool read(T* out) {
        if (!have(sizeof(T))) return false;
        std::memcpy(out, p + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }
};

// Size in bytes of a fixed-size GGUF value type, 0 for string/array/unknown.
size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;   // u8, i8, bool
        case 2: case 3: return 2;           // u16, i16
        case 4: case 5: case 6: return 4;   // u32, i32, f32
        case 10: case 11: case 12: return 8;  // u64, i64, f64
        default: return 0;
    }
}

constexpr uint32_t kString = 8, kArray = 9;
constexpr uint64_t kMaxString = 1ull << 30;  // same cap as ggml's reader

bool skip_string(Cursor& c) {
    uint64_t n;
    return c.read(&n) && n <= kMaxString && c.skip(n);
}

}  // namespace

bool gguf_precheck(const void* data, size_t size, std::string* err) {
    auto fail = [&](const char* m) {
        if (err) *err = m;
        return false;
    };
    Cursor c{static_cast<const uint8_t*>(data), size};
    uint32_t version;
    int64_t n_tensors, n_kv;
    if (!c.have(4) || std::memcmp(c.p, "GGUF", 4) != 0) return fail("not a GGUF file (bad magic)");
    c.pos = 4;
    if (!c.read(&version) || (version != 2 && version != 3))
        return fail("unsupported GGUF version");
    if (!c.read(&n_tensors) || !c.read(&n_kv) || n_tensors < 0 || n_kv < 0)
        return fail("GGUF header is truncated or has a negative count");
    // Every key/value pair takes at least 13 bytes, so a count that cannot fit is
    // corrupt (and must not drive a long loop).
    if ((uint64_t)n_kv > (size - c.pos) / 13) return fail("GGUF metadata count exceeds the buffer size");

    for (int64_t i = 0; i < n_kv; ++i) {
        uint64_t klen;
        if (!c.read(&klen) || klen > kMaxString || !c.have(klen))
            return fail("GGUF metadata key is truncated");
        if (klen == 0) return fail("GGUF metadata key has an empty name");
        c.skip(klen);
        uint32_t type;
        if (!c.read(&type)) return fail("GGUF metadata is truncated");
        if (type == kString) {
            if (!skip_string(c)) return fail("GGUF string value is truncated");
        } else if (type == kArray) {
            uint32_t et;
            uint64_t n;
            if (!c.read(&et) || !c.read(&n)) return fail("GGUF array header is truncated");
            if (et == kString) {
                for (uint64_t j = 0; j < n; ++j)
                    if (!skip_string(c)) return fail("GGUF string array is truncated");
            } else {
                const size_t es = scalar_size(et);
                if (es == 0) return fail("GGUF array has an unsupported element type");
                if (n > size / es || !c.skip(n * es)) return fail("GGUF array is truncated");
            }
        } else {
            const size_t s = scalar_size(type);
            if (s == 0) return fail("GGUF metadata has an unknown value type");
            if (!c.skip(s)) return fail("GGUF metadata is truncated");
        }
    }
    return true;
}

}  // namespace ced
