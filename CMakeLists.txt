cmake_minimum_required(VERSION 3.14)

project("argeo-jjml"
VERSION 0.1.0
DESCRIPTION "Java bindings for GGML-based libraries"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/sdk/argeo-build")
find_package(ArgeoBuild)

set(A2_CATEGORY org.argeo.jjml)

a2_build_bundles(org.argeo.jjml)
add_subdirectory(jni)
