# Convenience Makefile based on default Argeo SDK conventions
include  sdk/argeo-build/cmake/default.mk

##
# Run make clean / all / install for the default CMake build.
# If system ggml and/or llama.cpp libraries are found they will be used to build the JNI bindings,
# otherwise the missing layer will be buit from the source submodule.
# Use make rebuild-force-to (see below) in order to force a local build. 
##

GGML_BLAS ?= OFF
GGML_VULKAN ?= OFF
GGML_CUDA ?= OFF
GGML_RPC ?= OFF

# To be used for "heavy" C++ development,
# that is when adding new capabilities and exploring upstream code:
# - Make sure the target binaries are built from the local sources submodules
# - Build the tools and examples, so that they can be browsed, debugged and hacked in IDE
rebuild-force-tp: clean-local
	echo CMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	
	mkdir -p $(BUILD_BASE)
	cmake -B $(BUILD_BASE) . \
		-DJJML_FORCE_BUILD_TP=ON \
		\
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DGGML_CCACHE=ON \
		-DBUILD_SHARED_LIBS=ON \
		-DCMAKE_SKIP_BUILD_RPATH=ON \
		\
		-DLLAMA_BUILD_COMMON=OFF \
		-DLLAMA_BUILD_TOOLS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		\
		-DGGML_NATIVE=OFF \
		-DGGML_CPU_ALL_VARIANTS=ON \
		-DGGML_BACKEND_DL=ON \
		\
		-DGGML_BLAS=$(GGML_BLAS) \
		-DGGML_BLAS_VENDOR=OpenBLAS \
		-DGGML_VULKAN=$(GGML_VULKAN) \
		-DGGML_CUDA=$(GGML_CUDA) \
		-DGGML_CUDA_FORCE_MMQ=ON \
		-DGGML_CUDA_FA_ALL_QUANTS=OFF \
		-DGGML_RPC=$(GGML_RPC) \
	
	cmake --build $(BUILD_BASE) -j $(shell nproc)

	@$(RM) $(TARGET_NATIVE_OUTPUT)/vulkan-shaders-gen

# Remove locally built libraries
clean-local: clean
	$(RM) -rf $(BUILD_BASE)
	echo $(TARGET_NATIVE_OUTPUT)
ifeq ($(MSYS_VERSION),0)
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/libggml*.so
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/libllama*.so
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/libJava_org_argeo_jjml_*.so
	@$(RM) $(TARGET_NATIVE_OUTPUT)/vulkan-shaders-gen
else
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/ggml*.dll
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/llama*.dll
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/Java_org_argeo_jjml_*.dll
endif

install-deps:
ifeq ($(MSYS_VERSION),0)
else
	pacman -S --needed git make mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-cmake
	pacman -S --needed mingw-w64-ucrt-x86_64-ccache
	# Vulkan
	pacman -S --needed mingw-w64-ucrt-x86_64-vulkan-devel mingw-w64-ucrt-x86_64-shaderc
endif
