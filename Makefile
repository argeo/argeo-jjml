# Convenience Makefile based on default Argeo SDK conventions
include  sdk/argeo-build/cmake/default.mk

jjml-force-tp:
	echo CMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	$(RM) -rf $(BUILD_BASE)
	mkdir -p $(BUILD_BASE)
	cmake -B $(BUILD_BASE) . \
		-DJJML_FORCE_BUILD_TP=ON \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DGGML_CCACHE=ON \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_SKIP_BUILD_RPATH=ON \
		-DLLAMA_BUILD_COMMON=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		\
		-DGGML_NATIVE=ON \
		-DGGML_CPU_ALL_VARIANTS=OFF \
		-DGGML_BACKEND_DL=OFF \
		\
		-DGGML_BLAS=OFF \
		-DGGML_BLAS_VENDOR=OpenBLAS \
		-DGGML_VULKAN=OFF \
		-DGGML_CUDA=OFF \
		-DGGML_CUDA_FORCE_MMQ=ON \
		-DGGML_CUDA_FA_ALL_QUANTS=OFF \
	
	cmake --build $(BUILD_BASE) -j $(shell nproc)
	
