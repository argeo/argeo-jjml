# Convenience Makefile based on default Argeo SDK conventions
include  sdk/argeo-build/cmake/default.mk

jjml-force-tp:
	$(RM) -rf $(BUILD_BASE)
	mkdir -p $(BUILD_BASE)
	# Debug builds are too slow to be usable. TODO find out why
	cmake -B $(BUILD_BASE) . \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_SKIP_BUILD_RPATH=ON \
		-DJJML_FORCE_BUILD_TP=ON \
		-DGGML_BACKEND_DL=ON \
		-DGGML_BLAS=ON \
		-DGGML_BLAS_VENDOR=OpenBLAS \
		-DGGML_VULKAN=ON \
		-DGGML_CUDA=ON \
		-DGGML_OPENCL=OFF \
		-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
	
	cmake --build $(BUILD_BASE) -j $(shell nproc)
	