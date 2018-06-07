# Note: # pip install cmake
cmake_minimum_required(VERSION 3.10)
project(WebNeural)

# ┌─────────────────────────────────────────────────┐
# │ Detect emscripten                               │
# └─────────────────────────────────────────────────┘
if(CMAKE_CXX_COMPILER MATCHES "/em\\+\\+(-[a-zA-Z0-9.])?$")
  set(Web true)
else()
  set(Web false)
endif()

# ┌─────────────────────────────────────────────────┐
# │ Print Information about the compilation system  │
# └─────────────────────────────────────────────────┘
message(" * Build type : " ${CMAKE_BUILD_TYPE})
message(" * C compiler: ${CMAKE_C_COMPILER}")
message(" * C++ compiler: ${CMAKE_CXX_COMPILER}")
message(" * System: " ${CMAKE_SYSTEM_NAME} )
message(" * Use Web : " ${Web})

# ┌─────────────────────────────────────────────────┐
# │ C++14                                           │
# └─────────────────────────────────────────────────┘
if (Web)
  string(APPEND CMAKE_CXX_FLAGS " --std=c++14")
else()
  set (CMAKE_CXX_STANDARD 14)
endif()

# ┌─────────────────────────────────────────────────┐
# │ Warning                                         │
# └─────────────────────────────────────────────────┘
string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror")

# ┌─────────────────────────────────────────────────┐
# │ Include the source                              │
# └─────────────────────────────────────────────────┘
enable_testing()
add_subdirectory(src)
