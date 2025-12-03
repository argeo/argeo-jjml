# Makefile based on default Argeo SDK conventions,
# used as a high-level driver for build, packaging, etc.
# Low-level build only uses CMake.
-include sdk.mk
include sdk/argeo-build/cmake/default.mk
include sdk/argeo-build/jpms.mk

A2_CATEGORY=org.argeo.jjml
TARGET_NATIVE_OUTPUT_GGML=$(TARGET_NATIVE_OUTPUT)/org.argeo.tp.ggml
TARGET_NATIVE_OUTPUT_JJML=$(TARGET_NATIVE_OUTPUT)/$(A2_CATEGORY)

MODULES=\
org.argeo.jjml

JLINK_NATIVE_JMODS=\
org.argeo.jjml.jni \
org.argeo.tp.ggml.libs \
org.argeo.tp.ggml.llm.libs

JLINK_RT_MODULES=java.base,jdk.jshell

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

# Activate various features via environment variables:
GGML_BLAS ?= OFF
GGML_VULKAN ?= OFF
GGML_CUDA ?= OFF
GGML_RPC ?= OFF
GGML_OPENMP ?= OFF
GGML_CCACHE ?= ON

LLAMA_BUILD_TOOLS ?= ON
JJML_FORCE_BUILD_LLAMA_GGML ?= OFF

ifneq (,$(VCIDEInstallDir))
MSVC_IDE_BASE=$(shell cygpath -m '$(VCIDEInstallDir)\\..')
else
MSVC_IDE_BASE=$(shell cygpath -m 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/')
endif
#MSVC_CMAKE_BASE="$(MSVC_BUILD_TOOLS)/Common7/IDE/CommonExtensions/Microsoft/CMake"
MSVC_CMAKE_BASE=$(MSVC_IDE_BASE)CommonExtensions/Microsoft/CMake
MSVC_CMAKE="$(MSVC_CMAKE_BASE)/CMake/bin/cmake.exe"

ifeq ($(MSYS_VERSION),0)
JJML_CMAKE ?= $(CMAKE)
LLAMA_CURL ?= ON
else
JJML_CMAKE ?= $(MSVC_CMAKE)
LLAMA_CURL ?= OFF

# To build with MinGW compiler
# JJML_CMAKE=cmake LLAMA_CURL=ON make -f tooling.mk clean-local rebuild-force-tp
# (and add runtime to path)
endif

rebuild-force-tp:
	echo CMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	
	$(JJML_CMAKE) -B $(BUILD_BASE) . \
		-DJJML_FORCE_BUILD_TP=ON \
		-DJJML_FORCE_BUILD_LLAMA_GGML=$(JJML_FORCE_BUILD_LLAMA_GGML) \
		-DA2_INSTALL_MODE=a2 \
		-DJAVA_HOME=$(JAVA_HOME) \
		\
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCMAKE_SKIP_BUILD_RPATH=ON \
		-DGGML_CCACHE=$(GGML_CCACHE) \
		\
		-DLLAMA_CURL=$(LLAMA_CURL) \
		-DLLAMA_BUILD_COMMON=$(LLAMA_BUILD_TOOLS) \
		-DLLAMA_BUILD_TOOLS=$(LLAMA_BUILD_TOOLS) \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		\
		-DGGML_NATIVE=OFF \
		-DGGML_CPU_ALL_VARIANTS=ON \
		-DGGML_BACKEND_DL=ON \
		\
		-DGGML_OPENMP=$(GGML_OPENMP) \
		-DGGML_BLAS=$(GGML_BLAS) \
		-DGGML_BLAS_VENDOR=OpenBLAS \
		-DGGML_VULKAN=$(GGML_VULKAN) \
		-DGGML_CUDA=$(GGML_CUDA) \
		-DGGML_CUDA_FORCE_MMQ=ON \
		-DGGML_CUDA_FA_ALL_QUANTS=OFF \
		-DGGML_RPC=$(GGML_RPC) \
	
	$(JJML_CMAKE) --build $(BUILD_BASE) --config $(CMAKE_BUILD_TYPE) -j $(shell nproc)
	ln -f -r -s $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)*$(shlib_suffix) $(TARGET_NATIVE_OUTPUT)
	ln -f -r -s $(TARGET_NATIVE_OUTPUT_JJML)/$(shlib_prefix)*$(shlib_suffix) $(TARGET_NATIVE_OUTPUT)
	@$(RM) $(TARGET_NATIVE_OUTPUT_GGML)/vulkan-shaders-gen*

# Remove locally built libraries
clean-local:
	$(RM) -r $(A2_OUTPUT)/org.argeo.jjml
	$(RM) -r $(BUILD_BASE)
	$(RM) -r $(TARGET_NATIVE_OUTPUT_JJML)
	$(RM) -r $(TARGET_NATIVE_OUTPUT_GGML)
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/$(shlib_prefix)ggml*$(shlib_suffix)
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/$(shlib_prefix)llama*$(shlib_suffix)
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/$(shlib_prefix)Java_org_argeo_jjml_*$(shlib_suffix)
	@$(RM) -v $(TARGET_NATIVE_OUTPUT)/$(shlib_prefix)Java_org_argeo_jjml_*$(shlib_suffix).*

#
# DOC
#
doc-api:
	$(JAVA_HOME)/bin/javadoc -Xdoclint:none \
	 -d doc/reference/api \
	 -sourcepath org.argeo.jjml/src \
	 -subpackages org

#
# BUILD ENVIRONMENT
#

install-deps:
ifeq ($(MSYS_VERSION),0)
else
	pacman -S --needed git make mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-cmake
	pacman -S --needed mingw-w64-ucrt-x86_64-ccache
	# Vulkan
	pacman -S --needed mingw-w64-ucrt-x86_64-vulkan-devel mingw-w64-ucrt-x86_64-shaderc
endif

#
# PACKAGING
#
JMOD_JJML=org.argeo.jjml
JMOD_JJML_JNI=org.argeo.jjml.jni
JMOD_GGML=org.argeo.tp.ggml.libs
JMOD_GGML_LLM=org.argeo.tp.ggml.llm.libs

#JJML_JMODS ?= $(JMOD_JJML),$(JMOD_JJML_JNI),$(JMOD_GGML),$(JMOD_GGML_LLM)

#RT_JJML ?= rt-jjml
#RT_JJML_DIR = $(BUILD_BASE)/$(RT_JJML)-$(JLINK_JAVA_RELEASE)-$(JLINK_JVM_VARIANT)-$(TARGET_DEB_ARCH)
#RT_JJML_JMODS ?= java.base,jdk.compiler,jdk.jlink,jdk.jartool,jdk.jshell

#JDK_JJML ?= jdk-jjml-$(JLINK_JAVA_RELEASE)-$(JLINK_JVM_VARIANT)
#JDK_JJML_ARTIFACT = $(JDK_JJML)-$(TARGET_NATIVE_CATEGORY_PREFIX)
#JDK_JJML_DIR = $(BUILD_BASE)/$(JDK_JJML_ARTIFACT)

msvc-release:
	$(MSVC_CMAKE) \
		-B "$(BUILD_BASE)" \
		-DJJML_FORCE_BUILD_TP=ON \
		-DA2_INSTALL_MODE=a2 \
		-DJAVA_HOME="$(JAVA_HOME)" \
		\
		-DCMAKE_LIBRARY_ARCHITECTURE=x86_64-win32-default \
		-DLLAMA_BUILD_COMMON=ON \
		-DLLAMA_BUILD_TOOLS=ON \
		-DLLAMA_CURL=OFF \
		-DGGML_NATIVE=OFF \
		-DGGML_CPU_ALL_VARIANTS=ON \
		-DGGML_OPENMP=ON \
		-DGGML_BACKEND_DL=ON \
		-DGGML_VULKAN=OFF \
		"$(SDK_SRC_BASE)"

	$(MSVC_CMAKE) --build $(BUILD_BASE) --config Release -j $(shell nproc)
	ln -f -r -s $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)*$(shlib_suffix) $(TARGET_NATIVE_OUTPUT)
	ln -f -r -s $(TARGET_NATIVE_OUTPUT_JJML)/$(shlib_prefix)*$(shlib_suffix) $(TARGET_NATIVE_OUTPUT)

jmod-jjml: a2-prepare-output
	$(call a2_jmod_prepare_output,$(JMOD_JJML))
	$(COPY) COPYING.LESSER NOTICE $(JMODS_BASE)/$(JMOD_JJML)/legal
	
	# examples
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/man/examples
	$(COPY) -v sdk/jbin/*.java $(JMODS_BASE)/$(JMOD_JJML)/man/examples

	$(call a2_jmod_create,$(JMOD_JJML))
	# list content
	#$(JLINK_HOME)/bin/jmod list $(JLINK_A2_JMODS)/$(JMOD_JJML).jmod

jmod-jjml-jni: a2-prepare-output
	$(call a2_jmod_prepare_output,$(JMOD_JJML_JNI))
	$(COPY) COPYING.LESSER NOTICE $(JMODS_BASE)/$(JMOD_JJML_JNI)/legal

	$(COPY) $(TARGET_NATIVE_OUTPUT_JJML)/$(shlib_prefix)Java_org_argeo_jjml*$(shlib_suffix) \
	 $(JMODS_BASE)/$(JMOD_JJML_JNI)/lib

	$(call a2_jmod_bare_module,$(JMOD_JJML_JNI))
	$(call a2_jmod_create_native,$(JMOD_JJML_JNI))

jmod-ggml-libs: a2-prepare-output
	$(call a2_jmod_prepare_output,$(JMOD_GGML))

	$(COPY) native/tp/ggml/include/*.h $(JMODS_BASE)/$(JMOD_GGML)/include
	$(COPY) native/tp/ggml/LICENSE native/tp/ggml/AUTHORS $(JMODS_BASE)/$(JMOD_GGML)/legal
	
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)ggml$(shlib_suffix) $(JMODS_BASE)/$(JMOD_GGML)/lib
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)ggml-base$(shlib_suffix) $(JMODS_BASE)/$(JMOD_GGML)/lib
ifeq ($(TARGET_OS),macos)
	# When GGML_BACKEND_DL=ON, *.so are generated,
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/libggml-cpu*.so $(JMODS_BASE)/$(JMOD_GGML)/lib
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/libggml-metal.so $(JMODS_BASE)/$(JMOD_GGML)/lib
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/libggml-blas.so $(JMODS_BASE)/$(JMOD_GGML)/lib
	#  otherwise *.dylib
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/libggml-cpu*.dylib $(JMODS_BASE)/$(JMOD_GGML)/lib
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/libggml-metal.dylib $(JMODS_BASE)/$(JMOD_GGML)/lib
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/libggml-blas.dylib $(JMODS_BASE)/$(JMOD_GGML)/lib
else
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)ggml-cpu*$(shlib_suffix) $(JMODS_BASE)/$(JMOD_GGML)/lib
	#$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)ggml-vulkan$(shlib_suffix) $(JMODS_BASE)/$(JMOD_GGML)/lib
	# MSVC linker libs
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)ggml.lib $(JMODS_BASE)/$(JMOD_GGML)/lib
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)ggml-base.lib $(JMODS_BASE)/$(JMOD_GGML)/lib
endif
	
	$(call a2_jmod_bare_module,$(JMOD_GGML))
	$(call a2_jmod_create_native,$(JMOD_GGML))

jmod-ggml-llm-libs: a2-prepare-output
	$(call a2_jmod_prepare_output,$(JMOD_GGML_LLM))

	$(COPY) native/tp/llama.cpp/include/*.h $(JMODS_BASE)/$(JMOD_GGML_LLM)/include
	$(COPY) native/tp/llama.cpp/LICENSE native/tp/llama.cpp/AUTHORS $(JMODS_BASE)/$(JMOD_GGML_LLM)/legal
	
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)llama$(shlib_suffix) $(JMODS_BASE)/$(JMOD_GGML_LLM)/lib
	# MSVC linker libs
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)llama.lib $(JMODS_BASE)/$(JMOD_GGML_LLM)/lib
	#$(COPY) $(BUILD_BASE)/bin/llama-cli* $(JMODS_BASE)/$(JMOD_GGML_LLM)/bin

ifeq ($(TARGET_OS),macos)
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/llama-cli* $(JMODS_BASE)/$(JMOD_GGML_LLM)/lib
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/llama-bench* $(JMODS_BASE)/$(JMOD_GGML_LLM)/lib
else
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/llama-cli* $(JMODS_BASE)/$(JMOD_GGML_LLM)/bin
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/llama-bench* $(JMODS_BASE)/$(JMOD_GGML_LLM)/bin
endif

	$(call a2_jmod_bare_module,$(JMOD_GGML_LLM))
	$(call a2_jmod_create_native,$(JMOD_GGML_LLM))

#
# DISTRIBUTABLE PACKAGES
#
package-jmods: jmod-jjml jmod-jjml-jni jmod-ggml-libs jmod-ggml-llm-libs

rt-jjml: package-jmods
	$(call a2_jlink_create_rt,rt-jjml)

jdk-jjml: package-jmods
	$(call a2_jlink_create_jdk,jdk-jjml)
	
zip-jdk-jjml: jdk-jjml
	# create archive
	cd $(BUILD_BASE) && zip -r -q \
	 $(JDK_JJML_ARTIFACT)-$(A2_LAYER_VERSION).zip \
	 $(shell basename $(JDK_JJML_DIR))
	#rm -rf $(JDK_JJML_DIR)


JDK_JJML_WIN_UPGRADE_ID=d87918b9-88e7-51fb-92d5-7186ca73314b
# FIXME make it portable on non-MSYS Windows
#JDK_JJML_WIN_UPGRADE_ID=$(shell uuidgen --sha1 --namespace $(ARGEO_ENTERPRISE_NUMBER_UUID) --name $(JDK_JJML))

msi-jdk-jjml:
	$(call a2_jpackage_create_pkg jdk-jjml,JDK $(JLINK_JAVA_RELEASE) with additional machine learning features,Argeo GmbH,$(JDK_JJML_WIN_UPGRADE_ID))

install-msi-jdk-jjml:
	msiexec /i "$(BUILD_BASE)/jdk-jjml-$(JLINK_SUFFIX)-$(A2_LAYER_VERSION).msi" /passive

pkg-jdk-jjml:
	$(call a2_jpackage_create_pkg jdk-jjml,JDK $(JLINK_JAVA_RELEASE) with additional machine learning features,Argeo GmbH)

install-pkg-jdk-jjml:
	$(call a2_jpackage_install_pkg jdk-jjml)
	
# Note: On Windows, use dumpbin.exe in order to find depedencies of a DLL
# (similar to ldd on Linux). E.g. "C:\Program Files (x86)\Microsoft Visual
# Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\
# dumpbin.exe" /DEPENDENTS llama.dll
