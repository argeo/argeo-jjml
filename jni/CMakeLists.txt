cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("argeo-jjml-jni" C CXX)
project("argeo-jjml-jni")
#set(SOVERSION 0)

# Java
find_package(JNI)

if(MINGW)
   # cmake ../../ -DCMAKE_INSTALL_PREFIX=$MINGW_PREFIX
   set(CMAKE_SHARED_LIBRARY_PREFIX "")
else()

# libggml as external library
find_library(GGML_LOCATION ggml)
message (STATUS "Found GGML library: ${GGML_LOCATION}") 
add_library(ggml SHARED IMPORTED GLOBAL)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${GGML_LOCATION})

# libllama as external library
find_library(LLAMA_LOCATION llama)
message (STATUS "Found LLAMA library: ${LLAMA_LOCATION}") 
add_library(llama SHARED IMPORTED GLOBAL)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${LLAMA_LOCATION})
target_link_libraries(llama INTERFACE ${GGML_LOCATION})

endif()


include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_subdirectory(org_argeo_jjml_llama)
