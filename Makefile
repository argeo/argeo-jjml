# Convenience Makefile based on default Argeo SDK conventions
include  sdk/argeo-build/cmake/default.mk

##
# Run make clean / all / install for the default CMake build.
# If system ggml and/or llama.cpp libraries are found,
# they will be used to build the JNI bindings,
# otherwise the missing layer will be buit from the source submodule.
# Use make rebuild-force-to (see below) in order to force a local build. 

# rebuild-force-to: To be used for "heavy" C++ development,
# that is when adding new capabilities and exploring upstream code. It allows:
# - to ensure the target binaries are built from the local sources submodules
# - to build the tools and examples, so that they can be browsed, debugged,
# and hacked in an IDE.

# Activate various features via environment varibales:
GGML_BLAS ?= OFF
GGML_VULKAN ?= OFF
GGML_CUDA ?= OFF
GGML_RPC ?= OFF

rebuild-force-tp: clean-local
	echo CMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	
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
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/Java_org_argeo_jjml_*.dll
endif

##
## BUILD ENVIRONMENT
##

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

COPY=cp -rv 
JMODS_BASE=$(BUILD_BASE)/jmods
A2_JMODS=$(A2_OUTPUT)/jmods

JMOD_JJML=org.argeo.jjml

# MSYS2 (Windows)
UCRT64_BASE=/ucrt64
JMOD_UCRT=org.argeo.ftw.ucrt

JLINK_HOME ?= $(JAVA_HOME)
JLINK_JMODS ?= $(JLINK_HOME)/jmods

ifeq ($(MSYS_VERSION),0)
JJML_JMODS ?= $(JMOD_JJML)
else
JJML_JMODS ?= $(JMOD_UCRT),$(JMOD_JJML)
endif	

RT_JJML ?= rt-jjml
RT_JJML_JMODS ?= java.base,java.net.http,jdk.compiler,jdk.jlink,jdk.jartool,jdk.jshell

JDK_JJML ?= jdk-jjml
# Note: replacing $${MODULES// /,} is bash specific
JDK_JJML_JMODS ?= $(shell . $(JLINK_HOME)/release && echo $${MODULES// /,})
JDK_JJML_JAVA_VERSION = $(shell . $(JLINK_HOME)/release && echo $$JAVA_VERSION)

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

standalone-release: clean-local
	cmake -B $(BUILD_BASE) . \
		-DJJML_FORCE_BUILD_TP=ON \
		-DGGML_CCACHE=ON \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_SKIP_BUILD_RPATH=ON \
		-DLLAMA_BUILD_COMMON=ON \
		-DLLAMA_BUILD_TOOLS=ON \
		-DLLAMA_CURL=ON \
		-DGGML_NATIVE=OFF \
		-DGGML_CPU_ALL_VARIANTS=ON \
		-DGGML_BACKEND_DL=ON	
	cmake --build $(BUILD_BASE) -j $(shell nproc)

jmod-jjml:
	mkdir -p $(A2_JMODS)
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/bin
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/lib
#	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/lib/$(JMOD_JJML)/jbin
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/include
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/legal/{ggml,llama.cpp}

	# headers
	$(COPY) native/tp/ggml/include/ggml.h native/tp/ggml/include/ggml-backend.h \
	 $(JMODS_BASE)/$(JMOD_JJML)/include
	$(COPY) native/tp/llama.cpp/include/*.h $(JMODS_BASE)/$(JMOD_JJML)/include
	
	# legal
	$(COPY) COPYING.LESSER NOTICE $(JMODS_BASE)/$(JMOD_JJML)/legal
	$(COPY) native/tp/ggml/LICENSE native/tp/ggml/AUTHORS \
	 $(JMODS_BASE)/$(JMOD_JJML)/legal/ggml
	$(COPY) native/tp/llama.cpp/LICENSE native/tp/llama.cpp/AUTHORS \
	 $(JMODS_BASE)/$(JMOD_JJML)/legal/llama.cpp
	
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
	
	$(COPY) $(BUILD_BASE)/bin/llama-cli.exe $(JMODS_BASE)/$(JMOD_JJML)/bin
	
	$(COPY) $(A2_OUTPUT)/lib/local/Java_org_argeo_jjml*.dll $(JMODS_BASE)/$(JMOD_JJML)/lib
endif
#	$(COPY) sdk/jbin/* $(JMODS_BASE)/$(JMOD_JJML)/lib/$(JMOD_JJML)/jbin

	$(RM) $(A2_JMODS)/$(JMOD_JJML).jmod
	$(JAVA_HOME)/bin/jmod create \
	 --class-path $(A2_OUTPUT)/org.argeo.jjml/org.argeo.jjml.0.1.jar \
	 --libs $(JMODS_BASE)/$(JMOD_JJML)/lib \
	 --cmds $(JMODS_BASE)/$(JMOD_JJML)/bin \
	 --header-files $(JMODS_BASE)/$(JMOD_JJML)/include \
	 --legal-notices $(JMODS_BASE)/$(JMOD_JJML)/legal \
	 $(A2_JMODS)/$(JMOD_JJML).jmod
	# list content
	#$(JAVA_HOME)/bin/jmod list $(A2_JMODS)/$(JMOD_JJML).jmod

rt-jjml: standalone-release jmod-ftw-ucrt jmod-jjml
	$(RM) -r $(BUILD_BASE)/$(RT_JJML)
	$(JLINK_HOME)/bin/jlink \
	 --module-path $(JLINK_JMODS):$(A2_JMODS) \
	 --add-modules $(RT_JJML_JMODS),$(JJML_JMODS) \
	 --output $(BUILD_BASE)/$(RT_JJML)
	
	mkdir -p $(BUILD_BASE)/$(RT_JJML)/jmods
	$(COPY) $(A2_JMODS)/$(JMOD_JJML).jmod $(BUILD_BASE)/$(RT_JJML)/jmods
ifeq ($(MSYS_VERSION),0)
else
	$(COPY) $(A2_JMODS)/$(JMOD_UCRT).jmod $(BUILD_BASE)/$(RT_JJML)/jmods
endif	

jdk-jjml: standalone-release jmod-ftw-ucrt jmod-jjml
	$(RM) -r $(BUILD_BASE)/$(JDK_JJML)
	$(JLINK_HOME)/bin/jlink \
	 --module-path $(JLINK_JMODS):$(A2_JMODS) \
	 --add-modules $(JDK_JJML_JMODS),$(JJML_JMODS) \
	 --output $(BUILD_BASE)/$(JDK_JJML)
	
	mkdir -p $(BUILD_BASE)/$(JDK_JJML)/src
	cp $(JLINK_HOME)/lib/src.zip $(BUILD_BASE)/$(JDK_JJML)/lib
	mkdir -p $(BUILD_BASE)/$(JDK_JJML)/src
	cp -r org.argeo.jjml/src $(BUILD_BASE)/$(JDK_JJML)/src/org.argeo.jjml
	cd $(BUILD_BASE)/$(JDK_JJML)/src \
	 && zip -q -ur $(BUILD_BASE)/$(JDK_JJML)/lib/src.zip *
	$(RM) -r $(BUILD_BASE)/$(JDK_JJML)/src
	
	mkdir -p $(BUILD_BASE)/$(RT_JJML)/jmods
	$(COPY) $(A2_JMODS)/$(JMOD_JJML).jmod $(BUILD_BASE)/$(RT_JJML)/jmods
ifeq ($(MSYS_VERSION),0)
else
	$(COPY) $(A2_JMODS)/$(JMOD_UCRT).jmod $(BUILD_BASE)/$(RT_JJML)/jmods
endif
	
	# create archive
	cd $(BUILD_BASE) && \
	 zip -r -q $(JDK_JJML)-$(JDK_JJML_JAVA_VERSION)-$(shell date +%F).zip $(JDK_JJML)
	rm -rf $(BUILD_BASE)/$(JDK_JJML)
	
	
# Note: On Windows, use dumpbin.exe in order to find depedencies of a DLL
# (similar to ldd on Linux). E.g. "C:\Program Files (x86)\Microsoft Visual
# Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\
# dumpbin.exe" /DEPENDENTS llama.dll
