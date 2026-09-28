#include "ced_runner.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>

namespace ced {

// Generous metadata context (no_alloc: data lives in allocator buffers). The
// fused classify graph (embed + 12 blocks + head) is well under 1k nodes.
static constexpr size_t kGraphSize = 8192;

struct Backend::Impl {
    ggml_backend_t       backend = nullptr;      // selected device (GPU or CPU)
    ggml_backend_t       cpu_fallback = nullptr; // GPU path only, for unsupported ops
    ggml_gallocr_t       galloc = nullptr;       // persistent, reused by every graph
    ggml_backend_sched_t sched = nullptr;        // created only if a graph needs it
};

static bool iequals(const std::string& a, const char* b) {
    if (!b) return false;
    size_t i = 0;
    for (; i < a.size() && b[i]; ++i)
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
            return false;
    return i == a.size() && b[i] == '\0';
}

Backend::Backend() : impl_(new Impl()) {
    const char* env = std::getenv("CED_DEVICE");
    const std::string want = env ? env : "";
    if (!iequals(want, "cpu")) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(dev);
            const char* name = ggml_backend_dev_name(dev);
            const bool selected = want.empty()
                ? (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                   type == GGML_BACKEND_DEVICE_TYPE_IGPU)
                : iequals(want, name);
            if (!selected) continue;
            impl_->backend = ggml_backend_dev_init(dev, nullptr);
            if (impl_->backend) {
                device_name_ = name ? name : "";
                is_cpu_ = type == GGML_BACKEND_DEVICE_TYPE_CPU;
                break;
            }
        }
        if (!want.empty() && !impl_->backend)
            std::fprintf(stderr, "ced: CED_DEVICE=%s not found, using CPU\n", want.c_str());
    }
    if (!impl_->backend) {
        impl_->backend = ggml_backend_cpu_init();
        device_name_ = "cpu";
        is_cpu_ = true;
    }
    if (!impl_->backend) {
        std::fprintf(stderr, "ced: backend init failed\n");
        return;
    }
    if (!is_cpu_) impl_->cpu_fallback = ggml_backend_cpu_init();
}

Backend::~Backend() {
    // Allocators before the backends they reference.
    if (impl_->sched) ggml_backend_sched_free(impl_->sched);
    if (impl_->galloc) ggml_gallocr_free(impl_->galloc);
    if (impl_->cpu_fallback) ggml_backend_free(impl_->cpu_fallback);
    if (impl_->backend) ggml_backend_free(impl_->backend);
    delete impl_;
}

bool Backend::ok() const { return impl_->backend != nullptr; }

ggml_backend_t Backend::handle() const { return impl_->backend; }

bool Backend::compute(int n_threads, const BuildFn& build,
                      std::vector<std::vector<float>>& outs) {
    if (!impl_->backend) return false;

    size_t mem = ggml_tensor_overhead() * (kGraphSize * 2) +
                 ggml_graph_overhead_custom(kGraphSize, false);
    struct ggml_init_params ip { mem, nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) return false;

    std::vector<GraphInput> inputs;
    std::vector<ggml_tensor*> capture = build(ctx, inputs);
    if (capture.empty()) {
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    // Mark captured tensors as outputs so the allocator does NOT reuse their
    // buffers for downstream nodes (intermediates like enc_norm/logits feed
    // later ops and would otherwise read back freed/overwritten memory).
    for (ggml_tensor* t : capture) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }

    // A GPU graph goes through the scheduler only when the device lacks a
    // kernel for one of its ops; otherwise it takes the gallocr path.
    bool use_sched = false;
    if (impl_->cpu_fallback) {
        const int n = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n && !use_sched; ++i)
            use_sched = !ggml_backend_supports_op(impl_->backend, ggml_graph_node(gf, i));
    }

    const int nt = n_threads > 0 ? n_threads : 4;
    bool alloc_ok = false;
    if (use_sched) {
        if (!impl_->sched) {
            ggml_backend_t backs[2] = {impl_->backend, impl_->cpu_fallback};
            impl_->sched = ggml_backend_sched_new(backs, nullptr, 2, kGraphSize,
                                                  /*parallel=*/false, /*op_offload=*/true);
        }
        if (impl_->sched) {
            ggml_backend_cpu_set_n_threads(impl_->cpu_fallback, nt);
            ggml_backend_sched_reset(impl_->sched);
            alloc_ok = ggml_backend_sched_alloc_graph(impl_->sched, gf);
        }
    } else {
        if (!impl_->galloc)
            impl_->galloc =
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
        alloc_ok = impl_->galloc && ggml_gallocr_alloc_graph(impl_->galloc, gf);
    }
    if (!alloc_ok) {
        std::fprintf(stderr, "ced: graph alloc failed\n");
        ggml_free(ctx);
        return false;
    }

    for (const GraphInput& in : inputs)
        ggml_backend_tensor_set(in.t, in.data, 0, in.nbytes);

    if (is_cpu_) ggml_backend_cpu_set_n_threads(impl_->backend, nt);
    const ggml_status st = use_sched ? ggml_backend_sched_graph_compute(impl_->sched, gf)
                                     : ggml_backend_graph_compute(impl_->backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ced: graph compute failed\n");
        ggml_free(ctx);
        return false;
    }

    outs.resize(capture.size());
    for (size_t i = 0; i < capture.size(); ++i) {
        size_t n = (size_t)ggml_nelements(capture[i]);
        outs[i].resize(n);
        ggml_backend_tensor_get(capture[i], outs[i].data(), 0, n * sizeof(float));
    }
    ggml_free(ctx);
    return true;
}

}  // namespace ced
