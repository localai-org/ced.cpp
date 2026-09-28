set(CED_MODEL_GGUF    "${PROJECT_SOURCE_DIR}/models/ced-base-f32.gguf")
set(CED_BASELINE_GGUF "${PROJECT_SOURCE_DIR}/tests/fixtures/ced-base.baseline.gguf")

foreach(t blocks frontend quant capi)
  add_executable(test_${t} test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE ced)
  target_include_directories(test_${t} PRIVATE
      ${PROJECT_SOURCE_DIR}/src
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/third_party/ggml/include)
endforeach()
add_test(NAME capi COMMAND test_capi "${CED_MODEL_GGUF}" "${CED_BASELINE_GGUF}")

# Per-component parity (f32 reference path).
add_test(NAME blocks   COMMAND test_blocks   "${CED_MODEL_GGUF}" "${CED_BASELINE_GGUF}")
add_test(NAME frontend COMMAND test_frontend "${CED_MODEL_GGUF}" "${CED_BASELINE_GGUF}")

# Short-clip parity (T < target_length -> OW < 63): exercises the sliced
# time_pos_embed path. Regenerate the fixture with:
#   python scripts/gen_ced_baseline.py --model mispeech/ced-base \
#     --out tests/fixtures/ced-base-short.baseline.gguf --n-samples 64000
add_test(NAME frontend_short COMMAND test_frontend "${CED_MODEL_GGUF}"
         "${PROJECT_SOURCE_DIR}/tests/fixtures/ced-base-short.baseline.gguf")

# End-to-end quantization parity (probs close + top-5 labels unchanged).
add_test(NAME e2e_f32 COMMAND test_quant "${PROJECT_SOURCE_DIR}/models/ced-base-f32.gguf"  "${CED_BASELINE_GGUF}" 1e-3)
add_test(NAME e2e_f16 COMMAND test_quant "${PROJECT_SOURCE_DIR}/models/ced-base-f16.gguf"  "${CED_BASELINE_GGUF}" 5e-3)
add_test(NAME e2e_q8  COMMAND test_quant "${PROJECT_SOURCE_DIR}/models/ced-base-q8_0.gguf" "${CED_BASELINE_GGUF}" 3e-2)
