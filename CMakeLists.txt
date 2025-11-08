cmake_minimum_required(VERSION 3.14)

project("argeo-jjml"
DESCRIPTION "Enterprise-grade Java bindings for the ggml ecosystem"
)
set(A2_CATEGORY org.argeo.jjml)

if(NOT DEFINED A2_JAVA_RELEASE)
set(A2_JAVA_RELEASE 11)
endif()

# Can build with headless JDK
set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/sdk/argeo-build/cmake")
find_package(ArgeoBuild)

# JAVA
set(BUNDLES 
org.argeo.jjml
)

a2_build_bundles(${BUNDLES})

add_subdirectory(native)
