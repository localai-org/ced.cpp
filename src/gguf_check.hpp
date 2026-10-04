#pragma once
#include <cstddef>
#include <string>

namespace ced {

// Cheap structural check of a GGUF held in memory, run BEFORE the buffer is
// handed to ggml's reader. ggml validates sizes and bounds, but it still aborts
// the whole process (GGML_ASSERT) on a few well-formed-looking inputs, such as a
// metadata key with an empty name. A model buffer comes from the caller, so such
// input must become an error instead.
//
// Walks the header and the metadata key/value section with bounds checks on every
// read. Returns false and sets `err` on: a bad magic or version, an empty or
// oversized key, an unknown value type, a nested array, or a value that runs
// past the end of the buffer. It does not look at the tensor table: ggml reports
// those errors without aborting, and the loaders re-check every tensor range.
bool gguf_precheck(const void* data, size_t size, std::string* err);

}  // namespace ced
