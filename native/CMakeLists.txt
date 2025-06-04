#
# THIRD PARTIES
#

#set(CMAKE_FIND_DEBUG_MODE ON)

# Current Debian and Argeo paths
# TODO ask Debian to package llama.cpp headers
set(CMAKE_LIBRARY_PATH
/usr/lib/x86_64-linux-gnu/ggml
/usr/libexec/x86_64-linux-gnu/ggml
)
include_directories(
/usr/include/ggml
)

if(NOT JJML_FORCE_BUILD_TP)
	find_package(llama)
	if(NOT llama_FOUND)
		find_package(ggml)
	endif()
	find_library(GGML_LOCATION ggml)
	find_library(GGML_BASE_LOCATION ggml-base)
	find_library(LLAMA_LOCATION llama)
endif()

# ggml as external library
if(EXISTS ${GGML_LOCATION} AND EXISTS ${GGML_BASE_LOCATION} AND NOT JJML_FORCE_BUILD_TP)
#	if(MINGW)
#		add_library(ggml STATIC IMPORTED GLOBAL)
#	else()
#		add_library(ggml SHARED IMPORTED GLOBAL)
#	endif()
#	set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${GGML_LOCATION})
	message (STATUS "Found GGML library: ${GGML_LOCATION}")
	
#	find_library(GGML_BASE_LOCATION ggml-baseX)
#	if(EXISTS ${GGML_BASE_LOCATION})
#		message (STATUS "Found GGML Base library: ${GGML_BASE_LOCATION}")
#	else()
#		message (FATAL "GGML Base library not found")
#	endif()
else()
	## multiple backends libraries
	if(GGML_BACKEND_DL)
		set(GGML_NATIVE OFF)
		set(GGML_CPU_ALL_VARIANTS ON)
	endif()

	# copied from root of ggml, otherwise
	# standalone build of GGML with GGML_BACKEND_DL fails 
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})

	add_subdirectory(tp/ggml) # build locally
	message (STATUS "Will locally build ggml from submodule native/tp/ggml") 
endif()

# llama.cpp as external library
if(EXISTS ${LLAMA_LOCATION} AND NOT JJML_FORCE_BUILD_TP)
#	if(MINGW)
#		add_library(llama STATIC IMPORTED GLOBAL)
#	else()
#		add_library(llama SHARED IMPORTED GLOBAL)
#	endif()
#	set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${LLAMA_LOCATION})
#	target_link_libraries(llama INTERFACE ${GGML_LOCATION})
	message (STATUS "Found LLAMA library: ${LLAMA_LOCATION}") 
else()
	set(LLAMA_STANDALONE ON)
	add_subdirectory(tp/llama.cpp) # build locally
	set_target_properties(llama PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})
	# Required by MSYS
	set_target_properties(llama PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})
	message (STATUS "Will locally build llama.cpp from submodule native/tp/llama.cpp") 
endif()


#
# JJML NATIVE BUILD
#
add_subdirectory(org_argeo_jjml_ggml)
add_subdirectory(org_argeo_jjml_llama)
