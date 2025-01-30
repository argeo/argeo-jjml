# Convenience Makefile based on default Argeo SDK conventions
include  sdk/argeo-build/cmake/default.mk

jjml-force-tp:
	$(RM) -rf $(BUILD_BASE)
	mkdir -p $(BUILD_BASE)
	cmake -B $(BUILD_BASE) . \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DJJML_FORCE_BUILD_TP=OFF \
		-DGGML_BLAS=ON \
		-DGGML_BLAS_VENDOR=OpenBLAS \
		-DGGML_VULKAN=ON \
		-DGGML_CUDA=OFF \
		-DGGML_OPENCL=OFF \
		-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
		-DGGML_BACKEND_DL=ON
	cmake --build $(BUILD_BASE) -j $(shell nproc)
	