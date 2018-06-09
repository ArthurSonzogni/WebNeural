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
# │ OpenMP                                          │
# └─────────────────────────────────────────────────┘
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# ┌─────────────────────────────────────────────────┐
# │ MNIST reader                                    │
# └─────────────────────────────────────────────────┘
find_package(MNIST PATHS third_party/mnist)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR
     "MNIST loader could not be found.
     Please execute:
     git submodule init
     git submodule update")
 endif(NOT MNIST_FOUND)
include_directories(${MNIST_INCLUDE_DIR})

# ┌─────────────────────────────────────────────────┐
# │ Include the source                              │
# └─────────────────────────────────────────────────┘
enable_testing()
add_subdirectory(src)
