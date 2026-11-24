if (Web)
  file(READ style.css style)
  file(READ main.js script)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/figure.svg
    ${CMAKE_CURRENT_BINARY_DIR}/figure.svg
  )

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/index.html
    ${CMAKE_CURRENT_BINARY_DIR}/index.html
  )

  add_executable(main.js
    main.cpp
  )

  set_target_properties(main.js PROPERTIES
    OUTPUT_NAME "main"
    SUFFIX ".js"
  )

  target_link_libraries(main.js
    main
  )

  target_compile_options(main.js PRIVATE -O3)
  target_link_options(main.js PRIVATE
    -O3
    -sINITIAL_MEMORY=268435456
    -sALLOW_MEMORY_GROWTH=1
    "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/save.bin@save.bin"
    "SHELL:--preload-file ${MNIST_DATA_DIR}/t10k-images-idx3-ubyte@/data/t10k-images-idx3-ubyte"
    "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/empty@/data/t10k-labels-idx1-ubyte"
    "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/empty@/data/train-images-idx3-ubyte"
    "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/empty@/data/train-labels-idx1-ubyte"
    "-sEXPORTED_FUNCTIONS=_malloc,_LastInput,_LastOutput,_Train,_Predict,_LoadPretrainedModel"
    "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF64"
  )

  target_compile_definitions(main.js PRIVATE Web="${Web}")
  target_compile_definitions(main.js PRIVATE MNIST_DATA_LOCATION="./data/")
else (Web)
  add_executable(train main.cpp)
  target_link_libraries(train main)
  target_compile_definitions(train
    PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
endif(Web)
