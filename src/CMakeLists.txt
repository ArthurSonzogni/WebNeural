include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(main
  Image.cpp
  Image.hpp
  Optimizer.cpp
  Optimizer.hpp
  Tensor.cpp
  Tensor.hpp
  node/Bias.cpp
  node/Bias.hpp
  node/Convolution2D.cpp
  node/Convolution2D.hpp
  node/Dropout.cpp
  node/Dropout.hpp
  node/Input.cpp
  node/Input.hpp
  node/Linear.cpp
  node/Linear.hpp
  node/MaxPooling.cpp
  node/MaxPooling.hpp
  node/Node.cpp
  node/Node.hpp
  node/Relu.cpp
  node/Relu.hpp
  node/Sigmoid.cpp
  node/Sigmoid.hpp
  node/Softmax.cpp
  node/Softmax.hpp
  util.cpp
  util.hpp
)

# ┌─────────────────────────────────────────────────┐
# │ Test                                            │
# └─────────────────────────────────────────────────┘

# Note: For gtest, please follow:
# https://stackoverflow.com/questions/24295876/cmake-cannot-find-a-googletest-required-library
find_package(GTest REQUIRED)
find_package (Threads)
include_directories(${GTest_INCLUDE_DIRS})
add_executable(tests
  node/Convolution2DTest.cpp
  node/LinearTest.cpp
  node/ReluTest.cpp
  node/SoftmaxTest.cpp
  test/mnist_test.cpp
  )
target_link_libraries(tests
  main
  ${GTEST_BOTH_LIBRARIES}
  Threads::Threads
  )
target_compile_definitions(tests PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
gtest_discover_tests(tests)
add_test(tests tests)
