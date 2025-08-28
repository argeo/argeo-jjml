#
# THIRD PARTIES
#

# Debug
#set(CMAKE_FIND_DEBUG_MODE ON)

# We always want the shared libraries
set(BUILD_SHARED_LIBS ON)

# Argeo paths
set(CMAKE_LIBRARY_PATH
/usr/libexec/${CMAKE_LIBRARY_ARCHITECTURE}/ggml
)

# Debian official until libraries are stable
set(CMAKE_PREFIX_PATH
/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/ggml/cmake-private
/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/llama/cmake-private
)

set(TARGET_NATIVE_OUTPUT_GGML ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX}/org.argeo.tp.ggml)
set(TARGET_NATIVE_OUTPUT_LLAMA ${TARGET_NATIVE_OUTPUT_GGML})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_GGML})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_GGML})

if(NOT JJML_FORCE_BUILD_TP)
	find_package(llama QUIET)
	if(llama_FOUND)
		message(STATUS "Found llama and ggml packages: ${llama_LIBRARY} ${GGML_LIBRARY} ${GGML_BASE_LIBRARY}")
	else()
		find_library(llama_LIBRARY llama)
		
		find_package(ggml QUIET)
		if(ggml_FOUND)
			message(STATUS "Found GGML package: ${GGML_LIBRARY} ${GGML_BASE_LIBRARY}")
		else()
			find_library(GGML_LIBRARY ggml)
			find_library(GGML_BASE_LIBRARY ggml-base)
		endif()
	endif()
endif()

# ggml as external library
if(EXISTS ${GGML_LIBRARY})
	if(NOT ggml_FOUND)
		add_library(ggml SHARED IMPORTED GLOBAL)
		set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${GGML_LIBRARY})
		add_library(ggml-base SHARED IMPORTED GLOBAL)
		set_target_properties(ggml-base PROPERTIES IMPORTED_LOCATION ${GGML_BASE_LIBRARY})
		message(STATUS "Found ggml libraries: ${GGML_LIBRARY} ${GGML_BASE_LIBRARY}")
	endif()
	if(NOT DEFINED ${GGML_BACKEND_DL})
		include(CheckIncludeFileCXX)
		check_include_file_cxx(ggml-backend.h JJML_BACKEND_INCLUDE_FOUND)
		if(JJML_BACKEND_INCLUDE_FOUND)
			set(GGML_BACKEND_DL ON)
			message(STATUS "Found ggml-backend.h, enabling GGML_BACKEND_DL")
		endif()
	endif()
else()
	## multiple backends libraries
	if(NOT DEFINED ${GGML_BACKEND_DL})
		set(GGML_BACKEND_DL ON)
		message(STATUS "Enabling GGML_BACKEND_DL by default")
	endif()
	if(GGML_BACKEND_DL)
		set(GGML_CPU_ALL_VARIANTS ON)
	endif()

	add_subdirectory(tp/ggml) # build locally
	message(STATUS "Build ggml locally from submodule native/tp/ggml") 
endif()

# llama.cpp as external library
if(EXISTS "${llama_LIBRARY}")
	if(NOT llama_FOUND)
		add_library(llama SHARED IMPORTED GLOBAL)
		set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${llama_LIBRARY})
		target_link_libraries(llama INTERFACE ${GGML_LIBRARY})
	endif()
	message(STATUS "Found LLAMA library: ${llama_LIBRARY}") 
else()
	set(LLAMA_STANDALONE ON)
	add_subdirectory(tp/llama.cpp) # build locally
	if(MINGW)
		set_target_properties(llama PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_LLAMA})
	else()
		set_target_properties(llama PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_LLAMA})
	endif()
	message(STATUS "Build llama.cpp locally from submodule native/tp/llama.cpp") 
endif()

#
# JJML NATIVE BUILD
#
if(GGML_BACKEND_DL)
	message(STATUS "Build JNI libraries with support for dynamic ggml backends") 
endif()

message(STATUS "GGML_VERSION=${GGML_VERSION} LLAMA_VERSION=${LLAMA_VERSION}") 

set(CMAKE_CXX_STANDARD 17)
if (NOT MSVC)
	# force ISO-C++
    #add_compile_options(-pedantic-errors)
    add_compile_options(-pedantic)
endif()

add_subdirectory(org_argeo_jjml_ggml)
add_subdirectory(org_argeo_jjml_llm)
                      