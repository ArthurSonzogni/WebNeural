option(ADD_GOOGLE_ANALYTICS "Add google analytics script" ON)
if (ADD_GOOGLE_ANALYTICS)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/google-analytics.html google-analytics)
endif(ADD_GOOGLE_ANALYTICS)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/run_web_server.sh
  ${CMAKE_CURRENT_BINARY_DIR}/run_web_server.sh
)

file(READ "style.css" "style")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/index.html
  ${CMAKE_CURRENT_BINARY_DIR}/index.html
)

add_subdirectory(AutoEncoderInterpolation)
add_subdirectory(AutoEncoderInterpolationFashion)
add_subdirectory(WCGAN_Interpolation)
