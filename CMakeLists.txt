cmake_minimum_required(VERSION 3.18)
project(ced_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CED_BUILD_TESTS "Build ctest targets" OFF)
option(CED_BUILD_CLI   "Build ced-cli" ON)
option(CED_SHARED      "Build libced as shared" OFF)
option(CED_GGML_CUDA   "Forward GGML_CUDA" OFF)
option(CED_GGML_METAL  "Forward GGML_METAL" OFF)
option(CED_GGML_VULKAN "Forward GGML_VULKAN" OFF)
option(CED_GGML_HIP    "Forward GGML_HIP (ROCm)" OFF)

set(CED_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CED_TOP_LEVEL ON)
endif()

option(CED_EXTERNAL_DR_WAV "Use the embedding project's dr_wav implementation" OFF)

if(CED_TOP_LEVEL)
  set(GGML_CUDA   ${CED_GGML_CUDA}   CACHE BOOL "" FORCE)
  set(GGML_METAL  ${CED_GGML_METAL}  CACHE BOOL "" FORCE)
  set(GGML_VULKAN ${CED_GGML_VULKAN} CACHE BOOL "" FORCE)
  set(GGML_HIP    ${CED_GGML_HIP}    CACHE BOOL "" FORCE)

  # Free perf wins (per parakeet.cpp / rt-detr.cpp): -march=native + tinyBLAS.
  if(NOT DEFINED GGML_NATIVE)
    set(GGML_NATIVE ON CACHE BOOL "ggml: optimize for the current system" FORCE)
  endif()
  if(NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE ON CACHE BOOL "ggml: use LLAMAFILE (tinyBLAS SGEMM)" FORCE)
  endif()
endif()
if(NOT TARGET ggml)
  add_subdirectory(third_party/ggml)
endif()

set(CED_SRC
    src/model_loader.cpp
    src/ced_runner.cpp
    src/fft.cpp
    src/mel.cpp
    src/audio_io.cpp
    src/ced.cpp
    src/ced_capi.cpp)

if(CED_SHARED)
  add_library(ced SHARED ${CED_SRC})
  set_target_properties(ced PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(ced STATIC ${CED_SRC})
endif()
target_include_directories(ced PUBLIC include PRIVATE src ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(CED_EXTERNAL_DR_WAV)
  target_compile_definitions(ced PRIVATE CED_EXTERNAL_DR_WAV)
endif()
target_compile_definitions(ced PUBLIC $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
target_link_libraries(ced PUBLIC ggml)

if(CED_BUILD_CLI)
  add_subdirectory(examples/cli)
endif()
if(CED_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
