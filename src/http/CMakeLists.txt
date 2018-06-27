add_subdirectory(AutoEncoderInterpolation)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run_web_server.sh
  ${CMAKE_CURRENT_BINARY_DIR}/run_web_server.sh
)
