include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(main
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
)
#target_link_libraries(main profiler)

# ┌─────────────────────────────────────────────────┐
# │ Test                                            │
# └─────────────────────────────────────────────────┘

if (NOT Web)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed,-lprofiler")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no-as-needed,-lprofiler")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no-as-needed,-lprofiler")
endif()

if (NOT Web)

# Note: For gtest, please follow:
# https://stackoverflow.com/questions/24295876/cmake-cannot-find-a-googletest-required-library
find_package(GTest REQUIRED)
find_package (Threads)
include_directories(${GTest_INCLUDE_DIRS})

function(add_new_test test_name test_files)
  add_executable(${ARGV})
  target_link_libraries(${test_name}
    main
    ${GTEST_BOTH_LIBRARIES}
    Threads::Threads
  )
  target_compile_definitions(${test_name}
    PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
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


endif(NOT Web)

add_subdirectory(http)
