# Makefile based on default Argeo SDK conventions,
# used as a high-level driver for build, packaging, etc.
# Low-level build only uses CMake.
-include sdk.mk
include sdk/argeo-build/cmake/default.mk
include sdk/argeo-build/jpms.mk

TARGET_NATIVE_OUTPUT_GGML=$(TARGET_NATIVE_OUTPUT)/org.argeo.tp.ggml
TARGET_NATIVE_OUTPUT_JJML=$(TARGET_NATIVE_OUTPUT)/org.argeo.jjml

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
JMOD_GGML=org.argeo.tp.ggml
JMOD_GGML_LLM=org.argeo.tp.ggml.llm

JJML_JMODS ?= $(JMOD_JJML),$(JMOD_JJML_JNI),$(JMOD_GGML),$(JMOD_GGML_LLM)

RT_JJML ?= rt-jjml
RT_JJML_DIR = $(BUILD_BASE)/$(RT_JJML)-$(JLINK_JAVA_RELEASE)-$(JLINK_JVM_VARIANT)-$(TARGET_DEB_ARCH)
RT_JJML_JMODS ?= java.base,jdk.compiler,jdk.jlink,jdk.jartool,jdk.jshell

JDK_JJML ?= jdk-jjml-$(JLINK_JAVA_RELEASE)-$(JLINK_JVM_VARIANT)
JDK_JJML_ARTIFACT = $(JDK_JJML)-$(TARGET_NATIVE_CATEGORY_PREFIX)
JDK_JJML_DIR = $(BUILD_BASE)/$(JDK_JJML_ARTIFACT)

standalone-release: clean-local
	$(JJML_CMAKE) -B $(BUILD_BASE) . \
		-DJJML_FORCE_BUILD_TP=ON \
		-DA2_INSTALL_MODE=a2 \
		-DJAVA_HOME="$(JAVA_HOME)" \
		\
		-DGGML_CCACHE=ON \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_SKIP_BUILD_RPATH=ON \
		-DLLAMA_BUILD_COMMON=ON \
		-DLLAMA_BUILD_TOOLS=ON \
		-DLLAMA_CURL=OFF \
		-DGGML_NATIVE=OFF \
		-DGGML_CPU_ALL_VARIANTS=ON \
		-DGGML_BACKEND_DL=ON	
	$(CMAKE) --build $(BUILD_BASE) --config Release -j $(shell nproc)

#MSVC_BUILD_TOOLS="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"
#MSVC_ENV="/Common7/Tools/VsDevCmd.bat"

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
	$(RM) -r $(JMODS_BASE)/$(JMOD_JJML)
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/lib
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/legal
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML)/man/examples

	$(COPY) COPYING.LESSER NOTICE $(JMODS_BASE)/$(JMOD_JJML)/legal
	
	# examples
	$(COPY) -v sdk/jbin/*.java $(JMODS_BASE)/$(JMOD_JJML)/man/examples

	$(RM) $(A2_JMODS)/$(JMOD_JJML).jmod
	$(JLINK_HOME)/bin/jmod create \
	 --class-path $(A2_OUTPUT)/org.argeo.jjml/org.argeo.jjml.$(major).$(minor).jar \
	 --module-version $(A2_LAYER_VERSION) \
	 --man-pages $(JMODS_BASE)/$(JMOD_JJML)/man \
	 --legal-notices $(JMODS_BASE)/$(JMOD_JJML)/legal \
	 $(A2_JMODS)/$(JMOD_JJML).jmod
	# list content
	#$(JLINK_HOME)/bin/jmod list $(A2_JMODS)/$(JMOD_JJML).jmod

jmod-jjml-jni: a2-prepare-output
	$(RM) -r $(JMODS_BASE)/$(JMOD_JJML_JNI)
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML_JNI)/java
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML_JNI)/classes
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML_JNI)/lib
	mkdir -p $(JMODS_BASE)/$(JMOD_JJML_JNI)/legal

	$(COPY) COPYING.LESSER NOTICE $(JMODS_BASE)/$(JMOD_JJML_JNI)/legal

	$(COPY) $(TARGET_NATIVE_OUTPUT_JJML)/$(shlib_prefix)Java_org_argeo_jjml*$(shlib_suffix) \
	 $(JMODS_BASE)/$(JMOD_JJML_JNI)/lib

	echo "module $(JMOD_JJML_JNI) {}" > $(JMODS_BASE)/$(JMOD_JJML_JNI)/java/module-info.java
	$(JLINK_HOME)/bin/javac --release $(JLINK_JAVA_RELEASE) -d $(JMODS_BASE)/$(JMOD_JJML_JNI)/classes $(JMODS_BASE)/$(JMOD_JJML_JNI)/java/module-info.java

	$(RM) $(A2_JMODS)/$(JMOD_JJML_JNI)-$(TARGET_NATIVE_CATEGORY_PREFIX).jmod
	$(JLINK_HOME)/bin/jmod create \
	 --class-path $(JMODS_BASE)/$(JMOD_JJML_JNI)/classes \
	 --module-version $(A2_LAYER_VERSION) \
	 --target-platform $(JMOD_TARGET_PLATFORM) \
	 --libs $(JMODS_BASE)/$(JMOD_JJML_JNI)/lib \
	 --legal-notices $(JMODS_BASE)/$(JMOD_JJML_JNI)/legal \
	 $(A2_JMODS)/$(JMOD_JJML_JNI)-$(TARGET_NATIVE_CATEGORY_PREFIX).jmod
	# list content
	$(JLINK_HOME)/bin/jmod list $(A2_JMODS)/$(JMOD_JJML_JNI)-$(TARGET_NATIVE_CATEGORY_PREFIX).jmod

jmod-ggml: a2-prepare-output
	$(RM) -r $(JMODS_BASE)/$(JMOD_GGML)
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML)/java
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML)/classes
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML)/lib
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML)/include
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML)/legal

	$(COPY) native/tp/ggml/include/*.h $(JMODS_BASE)/$(JMOD_GGML)/include
	$(COPY) native/tp/ggml/LICENSE native/tp/ggml/AUTHORS $(JMODS_BASE)/$(JMOD_GGML)/legal
	
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)ggml$(shlib_suffix) $(JMODS_BASE)/$(JMOD_GGML)/lib
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)ggml-base$(shlib_suffix) $(JMODS_BASE)/$(JMOD_GGML)/lib
ifeq ($(TARGET_OS),macosx)
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
	echo "module $(JMOD_GGML) {}" > $(JMODS_BASE)/$(JMOD_GGML)/java/module-info.java
	$(JLINK_HOME)/bin/javac --release $(JLINK_JAVA_RELEASE) -d $(JMODS_BASE)/$(JMOD_GGML)/classes $(JMODS_BASE)/$(JMOD_GGML)/java/module-info.java

	$(RM) $(A2_JMODS)/$(JMOD_GGML).jmod
	$(JLINK_HOME)/bin/jmod create \
	 --class-path $(JMODS_BASE)/$(JMOD_GGML)/classes \
	 --target-platform $(JMOD_TARGET_PLATFORM) \
	 --libs $(JMODS_BASE)/$(JMOD_GGML)/lib \
	 --header-files $(JMODS_BASE)/$(JMOD_GGML)/include \
	 --legal-notices $(JMODS_BASE)/$(JMOD_GGML)/legal \
	 $(A2_JMODS)/$(JMOD_GGML).jmod
	# list content
	$(JLINK_HOME)/bin/jmod list $(A2_JMODS)/$(JMOD_GGML).jmod

jmod-ggml-llm: a2-prepare-output
	$(RM) -r $(JMODS_BASE)/$(JMOD_GGML_LLM)
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML_LLM)/java
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML_LLM)/classes
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML_LLM)/bin
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML_LLM)/lib
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML_LLM)/include
	mkdir -p $(JMODS_BASE)/$(JMOD_GGML_LLM)/legal

	$(COPY) native/tp/llama.cpp/include/*.h $(JMODS_BASE)/$(JMOD_GGML_LLM)/include
	$(COPY) native/tp/llama.cpp/LICENSE native/tp/llama.cpp/AUTHORS $(JMODS_BASE)/$(JMOD_GGML_LLM)/legal
	
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)llama$(shlib_suffix) $(JMODS_BASE)/$(JMOD_GGML_LLM)/lib
	# MSVC linker libs
	-$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/$(shlib_prefix)llama.lib $(JMODS_BASE)/$(JMOD_GGML_LLM)/lib
	#$(COPY) $(BUILD_BASE)/bin/llama-cli* $(JMODS_BASE)/$(JMOD_GGML_LLM)/bin
	$(COPY) $(TARGET_NATIVE_OUTPUT_GGML)/llama-cli* $(JMODS_BASE)/$(JMOD_GGML_LLM)/lib

# TODO add requires to ggml
	echo "module $(JMOD_GGML_LLM) {}" > $(JMODS_BASE)/$(JMOD_GGML_LLM)/java/module-info.java
	$(JLINK_HOME)/bin/javac --release $(JLINK_JAVA_RELEASE) -d $(JMODS_BASE)/$(JMOD_GGML_LLM)/classes $(JMODS_BASE)/$(JMOD_GGML_LLM)/java/module-info.java
	
	$(RM) $(A2_JMODS)/$(JMOD_GGML_LLM).jmod
	$(JLINK_HOME)/bin/jmod create \
	 --class-path $(JMODS_BASE)/$(JMOD_GGML_LLM)/classes \
	 --target-platform $(JMOD_TARGET_PLATFORM) \
	 --libs $(JMODS_BASE)/$(JMOD_GGML_LLM)/lib \
	 --cmds $(JMODS_BASE)/$(JMOD_GGML_LLM)/bin \
	 --header-files $(JMODS_BASE)/$(JMOD_GGML_LLM)/include \
	 --legal-notices $(JMODS_BASE)/$(JMOD_GGML_LLM)/legal \
	 $(A2_JMODS)/$(JMOD_GGML_LLM).jmod
	# list content
	$(JLINK_HOME)/bin/jmod list $(A2_JMODS)/$(JMOD_GGML_LLM).jmod

#
# DISTRIBUTABLE PACKAGES
#
rt-jjml: standalone-release jmod-jjml jmod-ggml jmod-ggml-llm
	$(RM) -r $(RT_JJML_DIR)
	$(JLINK_HOME)/bin/jlink \
	 --module-path "$(JLINK_JMODS)$(file_path_sep)$(A2_JMODS)" \
	 --add-modules $(RT_JJML_JMODS),$(JMOD_OS_LIBS),$(JJML_JMODS) \
	 --output "$(RT_JJML_DIR)"
	
	mkdir -p $(RT_JJML_DIR)/jmods
	$(COPY) $(A2_JMODS)/$(JMOD_JJML).jmod \
	 $(A2_JMODS)/$(JMOD_GGML)*.jmod \
	 $(A2_JMODS)/$(JMOD_OS_LIBS).jmod \
	 $(RT_JJML_DIR)/jmods

package-jmods: jmod-jjml jmod-jjml-jni jmod-ggml jmod-ggml-llm

jdk-jjml: package-jmods
	$(RM) -r $(JDK_JJML_DIR)
	$(JLINK_HOME)/bin/jlink \
	 --module-path "$(JLINK_JMODS)$(file_path_sep)$(A2_JMODS)" \
	 --add-modules $(JLINK_MODULES),$(JJML_JMODS) \
	 --output "$(JDK_JJML_DIR)"
	
	mkdir -p $(JDK_JJML_DIR)/src
	cp $(JLINK_HOME)/lib/src.zip $(JDK_JJML_DIR)/lib
	mkdir -p $(JDK_JJML_DIR)/src
	cp -r org.argeo.jjml/src $(JDK_JJML_DIR)/src/org.argeo.jjml
	cd $(JDK_JJML_DIR)/src \
	 && zip -q -ur $(JDK_JJML_DIR)/lib/src.zip *
	$(RM) -r $(JDK_JJML_DIR)/src
	
	mkdir -p $(JDK_JJML_DIR)/jmods
	$(COPY) $(A2_JMODS)/$(JMOD_JJML).jmod \
	 $(A2_JMODS)/$(JMOD_GGML)*.jmod \
	 $(JDK_JJML_DIR)/jmods

	mkdir -p $(JDK_JJML_DIR)/lib/a2/org.argeo.jjml
	$(COPY) $(A2_OUTPUT)/org.argeo.jjml/*.jar $(JDK_JJML_DIR)/lib/a2/org.argeo.jjml	
	
zip-jdk-jjml: jdk-jjml
	# create archive
	cd $(BUILD_BASE) && zip -r -q \
	 $(JDK_JJML_ARTIFACT)-$(A2_LAYER_VERSION).zip \
	 $(shell basename $(JDK_JJML_DIR))
	#rm -rf $(JDK_JJML_DIR)

ifneq (,$(shell which $(JLINK_HOME)/bin/jpackage))
msi-jdk-jjml:
	PATH=/usr/libexec/x86_64-win32-default/wix3:$(PATH) && \
	$(JLINK_HOME)/bin/jpackage \
	 --runtime-image $(JDK_JJML_DIR) \
	 --type msi \
	 --name $(JDK_JJML) \
	 --app-version $(A2_LAYER_VERSION) \
	 --dest $(BUILD_BASE) \
	 --description "JDK $(JLINK_JAVA_RELEASE) with additional machine learning features" \
	 --vendor "Argeo GmbH" \
	 --license-file "$(SDK_SRC_BASE)/NOTICE" \
	 --win-dir-chooser \
	 --win-per-user-install \
	 --win-upgrade-uuid $(shell uuidgen --sha1 --namespace $(ARGEO_ENTERPRISE_NUMBER_UUID) --name $(JDK_JJML)) \
	 --install-dir "$(JDK_JJML)" \
	
	mv $(BUILD_BASE)/$(JDK_JJML)-$(A2_LAYER_VERSION).msi \
	 $(BUILD_BASE)/$(JDK_JJML_ARTIFACT)-$(A2_LAYER_VERSION).msi

pkg-jdk-jjml:
	# .pkg format does not support versions with more than 3 components
	$(JLINK_HOME)/bin/jpackage \
	 --runtime-image $(JDK_JJML_DIR) \
	 --type pkg \
	 --name $(JDK_JJML) \
	 --app-version $(major).$(minor).$(micro) \
	 --dest $(BUILD_BASE) \
	 --description "JDK $(JLINK_JAVA_RELEASE) with additional machine learning features" \
	 --vendor "Argeo GmbH" \
	 --license-file "$(SDK_SRC_BASE)/NOTICE" \
	 --install-dir "/Library/Java/JavaVirtualMachines/$(JDK_JJML)" \
	
	ls -lash $(BUILD_BASE)/$(JDK_JJML)-*
	mv $(BUILD_BASE)/$(JDK_JJML)-$(major).$(minor).$(micro).pkg \
	 $(BUILD_BASE)/$(JDK_JJML_ARTIFACT)-$(A2_LAYER_VERSION).pkg

endif
	
# Note: On Windows, use dumpbin.exe in order to find depedencies of a DLL
# (similar to ldd on Linux). E.g. "C:\Program Files (x86)\Microsoft Visual
# Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\
# dumpbin.exe" /DEPENDENTS llama.dll
