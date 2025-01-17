#
# THIRD PARTIES
#
if(NOT MINGW)

# ggml as external library
find_library(GGML_LOCATION ggml)
if(EXISTS ${GGML_LOCATION} AND NOT JJML_FORCE_BUILD_TP)
	add_library(ggml SHARED IMPORTED GLOBAL)
	set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${GGML_LOCATION})
	message (STATUS "Found GGML library: ${GGML_LOCATION}") 
else()
	## multiple backends dlls
	if(GGML_BACKEND_DL)
		set(GGML_NATIVE OFF)
		set(GGML_CPU_ALL_VARIANTS ON)
	endif()

	# copied from root of ggml, otherwise
	# standalone build of GGML with GGML_BACKEND_DL fails 
	set(GGML_STANDALONE ON)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})

	add_subdirectory(tp/ggml) # build locally
	message (STATUS "Will build ggml library") 
endif()

# llama.cpp as external library
find_library(LLAMA_LOCATION llama)
if(EXISTS ${LLAMA_LOCATION} AND NOT JJML_FORCE_BUILD_TP)
	add_library(llama SHARED IMPORTED GLOBAL)
	set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${LLAMA_LOCATION})
	target_link_libraries(llama INTERFACE ${GGML_LOCATION})
	message (STATUS "Found LLAMA library: ${LLAMA_LOCATION}") 
else()
	set(LLAMA_STANDALONE ON)
	add_subdirectory(tp/llama.cpp) # build locally
	set_target_properties(llama PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX})
	message (STATUS "Will build llama.cpp library") 
endif()

endif() # MINGW

#
# JJML NATIVE BUILD
#
add_subdirectory(org_argeo_jjml_llama)
