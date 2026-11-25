set(AE_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/../AutoEncoderCommon")

if (Web)
  set(DEMO_TITLE "WebNeural: AutoEncoder Interpolation (Fashion-MNIST)")
  set(DEMO_DESC "Fashion-MNIST clothing autoencoder and latent-space interpolation in WebAssembly.")
  set(DEMO_HEADING "Convolutional AutoEncoder (Fashion-MNIST)")
  set(DEMO_SUBTITLE "Compress clothing items into 10 numbers, reconstruct them, and morph smoothly across the latent space.")
  set(DEMO_SCRIPT_NAME "autoencoder_fashion.js")

  file(READ "${AE_COMMON}/style.css" style)
  file(READ "${AE_COMMON}/main.js" script)

  configure_file(
    "${AE_COMMON}/figure.svg"
    "${CMAKE_CURRENT_BINARY_DIR}/figure.svg"
    COPYONLY
  )

  configure_file(
    "${AE_COMMON}/index.html.in"
    "${CMAKE_CURRENT_BINARY_DIR}/index.html"
  )

  add_executable(autoencoder_fashion.js
    "${AE_COMMON}/main.cpp"
  )

  set_target_properties(autoencoder_fashion.js PROPERTIES
    OUTPUT_NAME "autoencoder_fashion"
    SUFFIX ".js"
  )

  target_link_libraries(autoencoder_fashion.js
    main
  )

  target_compile_options(autoencoder_fashion.js PRIVATE -O3)
  target_link_options(autoencoder_fashion.js PRIVATE
    -O3
    -sINITIAL_MEMORY=268435456
    -sALLOW_MEMORY_GROWTH=1
    "SHELL:--preload-file ${CMAKE_CURRENT_SOURCE_DIR}/save.bin@save.bin"
    "SHELL:--preload-file ${MNIST_FASHION_DATA_DIR}/t10k-images-idx3-ubyte@/data/t10k-images-idx3-ubyte"
    "SHELL:--preload-file ${AE_COMMON}/empty@/data/t10k-labels-idx1-ubyte"
    "SHELL:--preload-file ${AE_COMMON}/empty@/data/train-images-idx3-ubyte"
    "SHELL:--preload-file ${AE_COMMON}/empty@/data/train-labels-idx1-ubyte"
    "-sEXPORTED_FUNCTIONS=_malloc,_LastInput,_LastOutput,_Train,_Predict,_LoadPretrainedModel,_ResetModelWeight"
    "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF64"
  )

  target_compile_definitions(autoencoder_fashion.js PRIVATE Web="${Web}")
  target_compile_definitions(autoencoder_fashion.js PRIVATE MNIST_DATA_LOCATION="./data/")
else (Web)
  add_executable(train_fashion "${AE_COMMON}/main.cpp")
  target_link_libraries(train_fashion main)
  target_compile_definitions(train_fashion
    PRIVATE MNIST_DATA_LOCATION="${MNIST_FASHION_DATA_DIR}")
endif(Web)
