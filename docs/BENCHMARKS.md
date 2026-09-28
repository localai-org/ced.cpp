# Benchmarks: ced.cpp vs PyTorch reference

End-to-end per-clip latency (mel frontend + ViT encoder + head → 527 probs),
CPU-only, same machine, same clip. ced.cpp matches the PyTorch reference
numerically (see [parity](../README.md)); this measures speed and memory.

- **Machine**: AMD Ryzen 9 9950X3D, CPU only (no GPU).
- **Model**: ced-base (86M params).
- **Clip**: 10.11 s (1012 mel frames) — the single-chunk path.
- **PyTorch**: `transformers` + `torchaudio`, f32 (no native CPU f16/int8).
- **ced.cpp**: ggml CPU, `-march=native` + tinyBLAS, built `Release`.
- 40 timed iterations after 8 warmup; mean reported. Reproduce with the commands
  at the bottom.

## Latency

| Implementation        | 4-thread mean | 4-thread RTF | 1-thread mean | peak RSS |
|-----------------------|--------------:|-------------:|--------------:|---------:|
| PyTorch (transformers, f32) | 158.8 ms | 64x | 399.1 ms | 717 MB |
| **ced.cpp f32** (same precision) | 126.6 ms | 80x  | 433.3 ms | 354 MB |
| **ced.cpp f16**       | **102.9 ms** | **98x** | 354.3 ms | 189 MB |
| **ced.cpp q8_0**      | 117.1 ms | 86x  | 410.3 ms | **111 MB** |

RTF = clip seconds / inference seconds (higher is faster; ~100x = 100 s of audio
classified per wall-second).

## Takeaways

- **Same precision (f32 vs f32)** is the apples-to-apples comparison: ced.cpp is
  **~1.25x faster (126.6 vs 158.8 ms) and uses 2x less memory (354 vs 717 MB)**
  than PyTorch, at the same numerical output.
- **The quantized configs go further** (near-lossless: identical top-5 tags):
  **f16 is the CPU sweet spot** at 102.9 ms/clip (~1.5x faster than PyTorch f32,
  ~100x realtime - tinyBLAS's f16 GEMM is the win), and **q8_0 drops to 111 MB**
  (~6.5x less than PyTorch) for a small dequant-overhead latency cost.
- **No Python/torch runtime is resident**, so peak RAM is much lower across the board.
- **Single thread**: ced.cpp f16/q8 still beat PyTorch, but f32 is marginally
  slower (PyTorch's oneDNN GEMM is strong single-threaded). On CPU, prefer f16.
- **Cold start**: `ced-cli classify` (model load + inference) completes in ~0.15 s
  wall; the PyTorch path pays multi-second `import torch`/`transformers` startup
  before the first inference — a large practical gap for serving and CLI use.
- **Parity holds throughout**: identical top-5 tags across all variants.

## GPU

The same build runs on any ggml GPU backend (see `CED_DEVICE` in the README).
Numbers below are mean `ced-cli bench` latency, 30 iterations after 3 warmup,
next to the CPU of the same machine at 4 threads. The 36 s clip is split into
four ~10 s chunks, so it measures the multi-chunk path.

| Device | Model | 6 s clip | 36 s clip | same host CPU, 36 s |
|---|---|--:|--:|--:|
| Apple M4, Metal | base f32  | 17.5 ms | 99.5 ms | 1151.7 ms |
| Apple M4, Metal | base q8_0 | 16.9 ms | 97.6 ms | 419.5 ms |
| Apple M4, Metal | tiny q8_0 | 5.7 ms  | 30.2 ms | 63.6 ms |
| Radeon 8060S, Vulkan (RADV) | base f32  | 14.7 ms | 70.6 ms | 446.4 ms |
| Radeon 8060S, Vulkan (RADV) | base q8_0 | 10.3 ms | 48.7 ms | 425.1 ms |
| Radeon 8060S, Vulkan (RADV) | tiny q8_0 | 7.0 ms  | 33.2 ms | 73.7 ms |
| NVIDIA GB10, CUDA 13 | base f32  | 11.3 ms | 62.6 ms | 2393.7 ms |
| NVIDIA GB10, CUDA 13 | base q8_0 | 10.5 ms | 61.8 ms | 2332.7 ms |
| NVIDIA GB10, CUDA 13 | tiny q8_0 | 9.9 ms  | 55.4 ms | 289.6 ms |

- **ced-base is 6x (Vulkan) to 12x (Metal) faster than the CPU of the same
  machine.** The GB10 CPU column is from a container build that is slow on
  CPU in general, so do not read the CUDA ratio as representative.
- **The GPU output keeps the CPU tags.** Over tiny and base at f32, f16 and
  q8_0 on both clips, the top-5 tags are the same on every backend. f32
  probabilities agree with the CPU to about 2e-4. Quantized models move more
  (up to ~1e-2 for tiny q8_0), because the CPU also quantizes the activations
  of q8_0 matmuls and the GPU backends do not.
- **Small models are bound by the host.** The mel frontend runs on the CPU,
  and for ced-tiny it is a large part of the GPU latency (on the GB10 about
  47 of the 55 ms on the 36 s clip). The GPU compute itself is a few
  milliseconds per chunk.

## Reproduce

```sh
# ced.cpp (per quant level, 4 and 1 threads)
ced-cli bench models/ced-base-f16.gguf  /tmp/ced_test.wav --iters 40 --warmup 8 --threads 4
ced-cli bench models/ced-base-q8_0.gguf /tmp/ced_test.wav --iters 40 --warmup 8 --threads 1

# PyTorch reference (same clip + methodology)
python scripts/bench_torch.py --model mispeech/ced-base --wav /tmp/ced_test.wav \
    --iters 40 --warmup 8 --threads 4
```
