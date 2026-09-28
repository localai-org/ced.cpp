add_executable(ced-cli main.cpp)
if(CED_EXTERNAL_DR_WAV)
  # libced.a was built without a dr_wav implementation (the embedding host
  # supplies one for its own binary), but ced-cli is a separate executable
  # that still needs the real symbols.
  target_sources(ced-cli PRIVATE dr_wav_impl.cpp)
endif()
target_link_libraries(ced-cli PRIVATE ced)
target_include_directories(ced-cli PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party)
