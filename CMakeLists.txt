cmake_minimum_required(VERSION 3.14)

project("argeo-jjml"
DESCRIPTION "Enterprise-grade Java bindings for the ggml ecosystem of machine learning libraries"
)
set(A2_CATEGORY org.argeo.jjml)

if(NOT DEFINED A2_JAVA_RELEASE)
set(A2_JAVA_RELEASE 11)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/sdk/argeo-build/cmake")
find_package(ArgeoBuild)

# JAVA
set(BUNDLES 
org.argeo.jjml
org.argeo.jjml.multimedia
)

a2_build_bundles(${BUNDLES})
add_dependencies(org.argeo.jjml.multimedia org.argeo.jjml)

if(NOT A2_BUILD_INDEP_ONLY)
add_subdirectory(native)
endif()
