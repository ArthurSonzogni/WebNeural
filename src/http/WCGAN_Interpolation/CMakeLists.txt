if (Web)
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

  set_target_properties(wgan_interpolation.js PROPERTIES
    OUTPUT_NAME "wgan_interpolation"
    SUFFIX ".js"
  )

  target_link_libraries(wgan_interpolation.js
    main
  )

  target_compile_options(wgan_interpolation.js PRIVATE -O3)
  target_link_options(wgan_interpolation.js PRIVATE
    -O3
    -sINITIAL_MEMORY=268435456
    -sALLOW_MEMORY_GROWTH=1
    "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/save.bin@save.bin"
    "SHELL:--preload-file ${MNIST_DATA_DIR}/t10k-images-idx3-ubyte@/data/t10k-images-idx3-ubyte"
    "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/empty@/data/train-images-idx3-ubyte"
    "-sEXPORTED_FUNCTIONS=_malloc,_Predict,_LoadPretrainedModel,_Train,_ResetModelWeight"
    "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF64"
  )

  target_compile_definitions(wgan_interpolation.js PRIVATE Web="${Web}")
  target_compile_definitions(wgan_interpolation.js PRIVATE MNIST_DATA_LOCATION="./data/")
else (Web)
  add_executable(train_wgan_interpolation main.cpp)
  target_link_libraries(train_wgan_interpolation main)
  target_compile_definitions(train_wgan_interpolation
    PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
endif(Web)
