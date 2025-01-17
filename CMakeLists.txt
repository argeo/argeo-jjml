cmake_minimum_required(VERSION 3.14)

project("argeo-jjml"
DESCRIPTION "Java bindings for GGML-based libraries"
VERSION 0.3.0
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

add_subdirectory(native)
