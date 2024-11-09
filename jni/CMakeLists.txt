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

# org_argeo_jjml_llama
add_library(org_argeo_jjml_llama SHARED 
	org_argeo_jjml_llama/org_argeo_jjml_llama_.cpp
	org_argeo_jjml_llama/jjml_llama.cpp
	org_argeo_jjml_llama/jjml_llama_vocabulary.cpp
	org_argeo_jjml_llama/jjml_llama_model.cpp
	org_argeo_jjml_llama/jjml_llama_context.cpp
	org_argeo_jjml_llama/jjml_llama_sampling.cpp
	org_argeo_jjml_llama/jjml_llama_batch.cpp
	org_argeo_jjml_llama/jjml_llama_embedding.cpp
)
set_target_properties(org_argeo_jjml_llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(org_argeo_jjml_llama PRIVATE ${JNI_INCLUDE_DIRS} include org_argeo_jjml_llama )
target_compile_features(org_argeo_jjml_llama PRIVATE cxx_std_11)
target_link_libraries(org_argeo_jjml_llama PRIVATE ggml llama)

set_target_properties(org_argeo_jjml_llama PROPERTIES 
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../../a2/${A2_TARGET_ARCH_CATEGORY_PREFIX}/${A2_CATEGORY}")

# MSYS2
#link_directories("/ucrt64/lib/")
#target_link_libraries(org_argeo_jjml_llama PRIVATE libggml libllama)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS org_argeo_jjml_llama LIBRARY)
