cmake_minimum_required(VERSION 3.14)

project("argeo-jjml"
DESCRIPTION "Java bindings for GGML-based libraries"
VERSION 0.1.0
)
set(A2_JAVA_RELEASE 11)
set(A2_CATEGORY org.argeo.jjml)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/sdk/argeo-build/cmake")
find_package(ArgeoBuild)

# JAVA
set(BUNDLES 
org.argeo.jjml
)

a2_build_bundles(${BUNDLES})

# NATIVE
if(NOT MINGW)

# libggml as external library
find_library(GGML_LOCATION ggml)
if(EXISTS ${GGML_LOCATION})
add_library(ggml SHARED IMPORTED GLOBAL)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${GGML_LOCATION})
message (STATUS "Found GGML library: ${GGML_LOCATION}") 
endif()

# libllama as external library
find_library(LLAMA_LOCATION llama)
if(EXISTS ${LLAMA_LOCATION})
add_library(llama SHARED IMPORTED GLOBAL)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${LLAMA_LOCATION})
target_link_libraries(llama INTERFACE ${GGML_LOCATION})
message (STATUS "Found LLAMA library: ${LLAMA_LOCATION}") 
else()
# build llama.cpp locally
add_subdirectory(sdk/tp/native)
set_target_properties(ggml PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})
set_target_properties(llama PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})
endif()

endif()

add_subdirectory(native)
