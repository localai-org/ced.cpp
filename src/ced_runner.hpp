#pragma once
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

namespace ced {

// An input leaf to be filled with host data AFTER the graph is allocated (the
// allocator decides the final address, so data is set post-alloc).
struct GraphInput {
    ggml_tensor* t = nullptr;
    const void*  data = nullptr;
    size_t       nbytes = 0;
};

// build() creates input leaves (registering each in `inputs`), builds the graph,
// and returns the tensors to capture. Backend::compute allocates the graph,
// uploads the inputs, computes, and reads each captured tensor's f32 data into
// `outs` (parallel to the returned vector).
using BuildFn =
    std::function<std::vector<ggml_tensor*>(ggml_context*, std::vector<GraphInput>&)>;

// Persistent compute backend + reusable graph allocator, one per loaded model.
//
// Device: CED_DEVICE names a registry device ("cpu", "CUDA0", "Vulkan0",
// "Metal", case-insensitive); unset picks the first GPU / integrated GPU and
// falls back to CPU.
//
// Every graph runs through ONE persistent ggml_gallocr, so the compute buffer
// is kept across calls instead of being allocated and freed per graph. On a GPU
// device, a graph that contains an op the device has no kernel for is routed
// through ggml_backend_sched with a CPU fallback instead; graphs the device
// fully supports stay on the gallocr path.
class Backend {
public:
    Backend();
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    bool ok() const;
    bool is_cpu() const { return is_cpu_; }
    const std::string& device_name() const { return device_name_; }
    ggml_backend_t handle() const;

    bool compute(int n_threads, const BuildFn& build,
                 std::vector<std::vector<float>>& outs);

private:
    struct Impl;
    Impl* impl_;
    bool is_cpu_ = true;
    std::string device_name_ = "cpu";
};

}  // namespace ced
