if (Web)
  string(APPEND CMAKE_CXX_FLAGS " -O3")
  #string(APPEND CMAKE_CXX_FLAGS " -s ALLOW_MEMORY_GROWTH=1")
  #string(APPEND CMAKE_CXX_FLAGS " -s TOTAL_MEMORY=1073741824")
  string(APPEND CMAKE_CXX_FLAGS " -s TOTAL_MEMORY=268435456")
  string(APPEND CMAKE_CXX_FLAGS " -s DEMANGLE_SUPPORT=1")
  string(APPEND CMAKE_CXX_FLAGS " --js-opts 3")
  string(APPEND CMAKE_CXX_FLAGS " --bind")
  string(APPEND CMAKE_CXX_FLAGS " --llvm-lto 3")
  string(APPEND CMAKE_CXX_FLAGS " --llvm-opts 3")
  string(APPEND CMAKE_CXX_FLAGS " -s ASSERTIONS=2")
  string(APPEND CMAKE_CXX_FLAGS " --closure 1")
  string(APPEND CMAKE_CXX_FLAGS " --no-heap-copy")
  string(APPEND CMAKE_CXX_FLAGS " --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/save.bin@save.bin")
 string(APPEND CMAKE_CXX_FLAGS " -s EXPORTED_FUNCTIONS='[\"_Predict\", \"_LoadPretrainedModel\", \"_Train\", \"_ResetModelWeight\"]'")
  string(APPEND CMAKE_CXX_FLAGS " -s EXTRA_EXPORTED_RUNTIME_METHODS='[\"ccall\",\"cwrap\"]'")
  
  string(APPEND CMAKE_CXX_FLAGS " --preload-file ${MNIST_DATA_DIR}/t10k-images-idx3-ubyte@/data/t10k-images-idx3-ubyte")
  #string(APPEND CMAKE_CXX_FLAGS " --preload-file ${MNIST_DATA_DIR}/train-images-idx3-ubyte@/data/train-images-idx3-ubyte")
  #string(APPEND CMAKE_CXX_FLAGS " --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/empty@/data/t10k-images-idx3-ubyte")
  string(APPEND CMAKE_CXX_FLAGS " --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/empty@/data/train-images-idx3-ubyte")

  file(READ style.css style)
  file(READ main.js script)
  file(READ snippet/headercpp.html headercpp)
  file(READ snippet/networkcpp.html networkcpp)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/figure.svg
    ${CMAKE_CURRENT_BINARY_DIR}/figure.svg
  )

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/index.html
    ${CMAKE_CURRENT_BINARY_DIR}/index.html
  )

  add_executable(wgan_interpolation.js
    main.cpp
  )

  target_link_libraries(wgan_interpolation.js
    main
  )

target_compile_definitions(wgan_interpolation.js PRIVATE Web="${Web}")
target_compile_definitions(wgan_interpolation.js PRIVATE MNIST_DATA_LOCATION="./data/")
else (Web)
  add_executable(train_wgan_interpolation main.cpp)
  target_link_libraries(train_wgan_interpolation main)
  target_compile_definitions(train_wgan_interpolation
    PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
endif(Web)
