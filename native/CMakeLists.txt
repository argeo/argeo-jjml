#
# THIRD PARTIES
#
#set(CMAKE_FIND_DEBUG_MODE ON)

# Current Debian and Argeo paths
# TODO ask Debian to package llama.cpp headers
set(CMAKE_LIBRARY_PATH
/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/ggml
/usr/libexec/${CMAKE_LIBRARY_ARCHITECTURE}/ggml
)
include_directories(
/usr/include/ggml
)


if(NOT JJML_FORCE_BUILD_TP)
	find_package(llama QUIET)
	if(NOT llama_FOUND)
		find_package(ggml QUIET)
	endif()
endif()
# we still search for libraries' locations in order to:
# - deal with the case when CMake config is not available
# - have a condition to trigger a local build
find_library(GGML_LOCATION ggml)
find_library(LLAMA_LOCATION llama)

# ggml as external library
if(EXISTS ${GGML_LOCATION} AND NOT JJML_FORCE_BUILD_TP)
	if(NOT ggml_FOUND)
		add_library(ggml SHARED IMPORTED GLOBAL)
		set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${GGML_LOCATION})
	endif()
	message (STATUS "Found GGML library: ${GGML_LOCATION}")
	include(CheckIncludeFileCXX)
	check_include_file_cxx(ggml-backend.h JJML_BACKEND_INCLUDE_FOUND)
	if(NOT DEFINED GGML_BACKEND_DL AND JJML_BACKEND_INCLUDE_FOUND)
		set(GGML_BACKEND_DL ON)
		message (STATUS "Found ggml-backend.h, enable GGML_BACKEND_DL")
	endif()
else()
	## multiple backends libraries
	if(NOT DEFINED GGML_BACKEND_DL)
		set(GGML_BACKEND_DL ON)
		set(BUILD_SHARED_LIBS ON)
	endif()
	if(GGML_BACKEND_DL)
		set(GGML_CPU_ALL_VARIANTS ON)
		message (STATUS "Enable GGML_BACKEND_DL by default")
	endif()

	# copied from root of ggml, otherwise
	# standalone build of GGML with GGML_BACKEND_DL fails 
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})

	add_subdirectory(tp/ggml) # build locally
	message (STATUS "Build ggml locally from submodule native/tp/ggml") 
endif()

# llama.cpp as external library
if(EXISTS ${LLAMA_LOCATION} AND NOT JJML_FORCE_BUILD_TP)
	if(NOT llama_FOUND)
		add_library(llama SHARED IMPORTED GLOBAL)
		set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${LLAMA_LOCATION})
		target_link_libraries(llama INTERFACE ${GGML_LOCATION})
	endif()
	message (STATUS "Found LLAMA library: ${LLAMA_LOCATION}") 
else()
	set(LLAMA_STANDALONE ON)
	add_subdirectory(tp/llama.cpp) # build locally
	set_target_properties(llama PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})
	# Required by MSYS
	set_target_properties(llama PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})
	message (STATUS "Build llama.cpp locally from submodule native/tp/llama.cpp") 
endif()

if(NOT DEFINED GGML_BACKEND_DL AND MINGW)
set(GGML_BACKEND_DL ON)
endif()

#
# JJML NATIVE BUILD
#
if(NOT CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(GGML_BACKEND_DL)
message (STATUS "Build JNI libraries with support for dynamic ggml backends") 
endif()

add_subdirectory(org_argeo_jjml_ggml)
add_subdirectory(org_argeo_jjml_llama)
