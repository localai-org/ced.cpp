// ced-cli is its own executable, never the embedding host's binary, so it
// needs a real dr_wav implementation even when the `ced` library is built
// with CED_EXTERNAL_DR_WAV (which only promises that *the host's* binary
// supplies one). Compiled into ced-cli only when that option is set; the
// non-embedded build already gets the implementation from src/audio_io.cpp
// via libced.a.
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
