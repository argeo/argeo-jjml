cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("argeo-jjml-jni" C CXX)
project("argeo-jjml-jni" VERSION 0.1.0)
set(SOVERSION 0)

# Java
find_package(JNI)

if (JNI_FOUND)
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
endif()

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

# org_argeo_jjml_llama
add_library(org_argeo_jjml_llama SHARED 
	org_argeo_jjml_llama/jjml_llama.cpp
	org_argeo_jjml_llama/jjml_llama_model.cpp
	org_argeo_jjml_llama/jjml_llama_batch.cpp
	org_argeo_jjml_llama/jjml_llama_embedding.cpp
)
set_target_properties(org_argeo_jjml_llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(org_argeo_jjml_llama PRIVATE org_argeo_jjml_llama ${JNI_INCLUDE_DIRS})
target_compile_features(org_argeo_jjml_llama PRIVATE cxx_std_11)

target_link_libraries(org_argeo_jjml_llama PRIVATE llama)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS org_argeo_jjml_llama LIBRARY)
