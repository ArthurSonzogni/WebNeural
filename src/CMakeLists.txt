include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(main
  algorithm/WCGAN.hpp
  algorithm/WCGAN.cpp
  Allocator.cpp
  Allocator.hpp
  Image.cpp
  Image.hpp
  LossFunction.cpp
  LossFunction.hpp
  Model.cpp
  Model.hpp
  PostUpdateFunction.cpp
  PostUpdateFunction.hpp
  Tensor.cpp
  Tensor.hpp
  node/BatchNormalization.cpp
  node/BatchNormalization.hpp
  node/Bias.cpp
  node/Bias.hpp
  node/BilinearUpsampling.cpp
  node/BilinearUpsampling.hpp
  node/Border.cpp
  node/Border.hpp
  node/Convolution2D.cpp
  node/Convolution2D.hpp
  node/Deconvolution2D.cpp
  node/Deconvolution2D.hpp
  node/Dropout.cpp
  node/Dropout.hpp
  node/Input.cpp
  node/Input.hpp
  node/LeakyRelu.cpp
  node/LeakyRelu.hpp
  node/Linear.cpp
  node/Linear.hpp
  node/MaxPooling.cpp
  node/MaxPooling.hpp
  node/Node.cpp
  node/Node.hpp
  node/Noise.cpp
  node/Noise.hpp
  node/Relu.cpp
  node/Relu.hpp
  node/Sigmoid.cpp
  node/Sigmoid.hpp
  node/Softmax.cpp
  node/Softmax.hpp
  node/Tanh.cpp
  node/Tanh.hpp
  util.cpp
  util.hpp
  util/stable_softmax.cpp
  util/stable_softmax.hpp
  ../third_party/stb_image/header.hpp
  ../third_party/stb_image/implementation.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(
    ../third_party/stb_image/implementation.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-lto;-Wno-stringop-overflow"
  )
endif()

if (NOT Web AND NOT MSVC)
  target_compile_options(main PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -ffp-contract=fast -fno-math-errno -fno-trapping-math>
  )
endif()

# ┌─────────────────────────────────────────────────┐
# │ ThinLTO                                         │
# └─────────────────────────────────────────────────┘
#include(CheckIPOSupported)
#check_ipo_supported()
#set_target_properties(main PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

add_library(sse4_1 INTERFACE IMPORTED)
if(MSVC)
    target_compile_options(sse4_1 INTERFACE /arch:SSE4.1)
else()
    target_compile_options(sse4_1 INTERFACE -msse4.1)
endif()

# ┌─────────────────────────────────────────────────┐
# │ Test                                            │
# └─────────────────────────────────────────────────┘

if (NOT Web)
  #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed,-lprofiler")
  #SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no-as-needed,-lprofiler")
  #SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no-as-needed,-lprofiler")
endif()

if (NOT Web)

enable_testing()

option(FETCHCONTENT_UPDATES_DISCONNECTED TRUE)
option(FETCHCONTENT_QUIET FALSE)
include(FetchContent)

FetchContent_Declare(googletest
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG        23ef29555ef4789f555f1ba8c51b4c52975f0907
  GIT_PROGRESS   TRUE
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  set(BUILD_GMOCK OFF CACHE INTERNAL "")
  set(INSTALL_GTEST OFF CACHE INTERNAL "")
  set(gtest_force_shared_crt ON CACHE INTERNAL "")
  add_subdirectory(
    ${googletest_SOURCE_DIR}
    ${googletest_BINARY_DIR}
    EXCLUDE_FROM_ALL
  )
endif()

include(GoogleTest)
function(add_new_test test_name test_files)
  add_executable(${ARGV})
  target_link_libraries(${test_name}
    PRIVATE main
    PRIVATE gtest
    PRIVATE gtest_main
    #PRIVATE Threads::Threads
  )
  target_compile_definitions(${test_name} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
  target_compile_definitions(${test_name} PRIVATE MNIST_FASHION_DATA_LOCATION="${MNIST_FASHION_DATA_DIR}")
  target_compile_definitions(${test_name} PRIVATE CIFAR_DATA_LOCATION="${CIFAR_DATA_DIR}")
  set_target_properties(main PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  gtest_discover_tests(${test_name})
  add_test(${test_name} ${test_name})
endfunction(add_new_test)

add_new_test(unit_tests
  node/Convolution2DTest.cpp
  node/Deconvolution2DTest.cpp
  node/LinearTest.cpp
  node/ReluTest.cpp
  node/SoftmaxTest.cpp
  ModelTest.cpp
)

add_new_test(mnist_tests 
  test/mnist_tests.cpp
)

add_new_test(gan_tests 
  test/gan_tests.cpp
)

add_new_test(wcgan_test 
  algorithm/WCGAN_test.cpp
)


endif(NOT Web)

add_subdirectory(http)
