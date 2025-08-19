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

##
## PACKAGING
##

# "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe" /DEPENDENTS ggml-cpu-icelake.dll

COPY=cp -rv 
UCRT64_BASE=/ucrt64
JMODS_BASE=$(BUILD_BASE)/jmods
A2_JMODS=$(A2_OUTPUT)/jmods

JMOD_UCRT=org.argeo.ftw.ucrt
JMOD_JJML=org.argeo.jjml

JLINK_HOME ?= $(JAVA_HOME)
JLINK_JMODS ?= $(JLINK_HOME)/jmods
RT_JJML ?= rt-jjml
ifeq ($(MSYS_VERSION),0)
RT_JJML_JMODS ?= java.base,java.net.http,jdk.compiler,jdk.jlink,jdk.jartool,jdk.jshell,$(JMOD_JJML)
else
RT_JJML_JMODS ?= java.base,java.net.http,jdk.compiler,jdk.jlink,jdk.jartool,jdk.jshell,$(JMOD_UCRT),$(JMOD_JJML)
endif	

jmod-ftw-ucrt:
ifeq ($(MSYS_VERSION),0)
else
	mkdir -p $(A2_JMODS)
	mkdir -p $(JMODS_BASE)/$(JMOD_UCRT)/java
	mkdir -p $(JMODS_BASE)/$(JMOD_UCRT)/classes
	mkdir -p $(JMODS_BASE)/$(JMOD_UCRT)/lib
	
	$(COPY) $(UCRT64_BASE)/bin/libgcc_s_seh-*.dll $(JMODS_BASE)/$(JMOD_UCRT)/lib
	$(COPY) $(UCRT64_BASE)/bin/libgomp-*.dll $(JMODS_BASE)/$(JMOD_UCRT)/lib
	$(COPY) $(UCRT64_BASE)/bin/libstdc++-*.dll $(JMODS_BASE)/$(JMOD_UCRT)/lib
	$(COPY) $(UCRT64_BASE)/bin/libwinpthread-*.dll $(JMODS_BASE)/$(JMOD_UCRT)/lib
	
	echo "module $(JMOD_UCRT) {}" > $(JMODS_BASE)/$(JMOD_UCRT)/java/module-info.java
	$(JAVA_HOME)/bin/javac --release 11 -d $(JMODS_BASE)/$(JMOD_UCRT)/classes $(JMODS_BASE)/$(JMOD_UCRT)/java/module-info.java
	
	$(RM) $(A2_JMODS)/$(JMOD_UCRT).jmod
	$(JAVA_HOME)/bin/jmod create \
	 --class-path $(JMODS_BASE)/$(JMOD_UCRT)/classes \
	 --libs $(JMODS_BASE)/$(JMOD_UCRT)/lib \
	 $(A2_JMODS)/$(JMOD_UCRT).jmod
endif

jmod-jjml:
	mkdir -p $(A2_JMODS)
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/lib
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/lib/$(JMOD_JJML)/jbin
ifeq ($(MSYS_VERSION),0)
	$(COPY) $(A2_OUTPUT)/lib/local/libggml.so $(JMODS_BASE)/$(JMOD_JJML)/lib
	$(COPY) $(A2_OUTPUT)/lib/local/libggml-base.so $(JMODS_BASE)/$(JMOD_JJML)/lib
	$(COPY) $(A2_OUTPUT)/lib/local/libggml-cpu-*.so $(JMODS_BASE)/$(JMOD_JJML)/lib
	$(COPY) $(A2_OUTPUT)/lib/local/libllama.so $(JMODS_BASE)/$(JMOD_JJML)/lib
	
	$(COPY) $(A2_OUTPUT)/lib/local/libJava_org_argeo_jjml*.so $(JMODS_BASE)/$(JMOD_JJML)/lib
else
	$(COPY) $(A2_OUTPUT)/lib/local/ggml.dll $(JMODS_BASE)/$(JMOD_JJML)/lib
	$(COPY) $(A2_OUTPUT)/lib/local/ggml-base.dll $(JMODS_BASE)/$(JMOD_JJML)/lib
	$(COPY) $(A2_OUTPUT)/lib/local/ggml-cpu-*.dll $(JMODS_BASE)/$(JMOD_JJML)/lib
#	$(COPY) $(A2_OUTPUT)/lib/local/ggml-vulkan.dll $(JMODS_BASE)/$(JMOD_JJML)/lib
	$(COPY) $(A2_OUTPUT)/lib/local/llama.dll $(JMODS_BASE)/$(JMOD_JJML)/lib
	
	$(COPY) $(A2_OUTPUT)/lib/local/Java_org_argeo_jjml*.dll $(JMODS_BASE)/$(JMOD_JJML)/lib
endif
	$(COPY) sdk/jbin/* $(JMODS_BASE)/$(JMOD_JJML)/lib/$(JMOD_JJML)/jbin

	$(RM) $(A2_JMODS)/$(JMOD_JJML).jmod
	$(JAVA_HOME)/bin/jmod create \
	 --class-path $(A2_OUTPUT)/org.argeo.jjml/org.argeo.jjml.0.1.jar \
	 --libs $(JMODS_BASE)/$(JMOD_JJML)/lib \
	 $(A2_JMODS)/$(JMOD_JJML).jmod
	
rt-jjml: jmod-ftw-ucrt jmod-jjml
	$(RM) -r $(BUILD_BASE)/$(RT_JJML)
	$(JLINK_HOME)/bin/jlink \
	 --module-path $(JLINK_JMODS):$(A2_JMODS) \
	 --add-modules $(RT_JJML_JMODS) \
	 --output $(BUILD_BASE)/$(RT_JJML)
	
	mkdir -p $(BUILD_BASE)/$(RT_JJML)/jmods
	$(COPY) $(A2_JMODS)/$(JMOD_JJML).jmod $(BUILD_BASE)/$(RT_JJML)/jmods
ifeq ($(MSYS_VERSION),0)
else
	$(COPY) $(A2_JMODS)/$(JMOD_UCRT).jmod $(BUILD_BASE)/$(RT_JJML)/jmods
endif	
