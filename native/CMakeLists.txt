#
# THIRD PARTY DETECTION
#
#set(CMAKE_FIND_DEBUG_MODE ON)

if(NOT JJML_FORCE_BUILD_TP)

if(NOT MSVC)
# Debian official until libraries are stable
set(CMAKE_PREFIX_PATH
/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/ggml/cmake-private
/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/llama/cmake-private
)
endif()

if(JJML_JDK_TP) # Use environment provided by a ggml-instrumented JDK
	set(CMAKE_LIBRARY_PATH ${JAVA_HOME}/bin ${JAVA_HOME}/lib)
	include_directories(${JAVA_HOME}/include SYSTEM)
endif()

# look for llama.cpp via CMake configs (preferred)
find_package(llama QUIET)
if(llama_FOUND)
	message(STATUS "Found llama.cpp and ggml packages: ${llama_LIBRARY} ${GGML_LIBRARY} ${GGML_BASE_LIBRARY}")
else()
	# look for llama.cpp via simple library search
	find_library(llama_LIBRARY llama)
	
	# look for ggml via CMake configs (preferred)
	find_package(ggml QUIET)
	if(ggml_FOUND)
		message(STATUS "Found ggml package: ${GGML_LIBRARY} ${GGML_BASE_LIBRARY}")
	else()
		# look for ggml via simple library search
		find_library(GGML_LIBRARY ggml)
		find_library(GGML_BASE_LIBRARY ggml-base)
	endif()
endif()
	
endif() # JJML_FORCE_BUILD_TP

if(EXISTS ${GGML_LIBRARY}) # ggml as external libraries
	if(NOT ggml_FOUND) # not CMake configured
		add_library(ggml SHARED IMPORTED GLOBAL)
		if(MSVC)
		set_target_properties(ggml PROPERTIES IMPORTED_IMPLIB ${GGML_LIBRARY})
		else()
		set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${GGML_LIBRARY})
		endif()
		
		add_library(ggml-base SHARED IMPORTED GLOBAL)
		if(MSVC)
		set_target_properties(ggml-base PROPERTIES IMPORTED_IMPLIB ${GGML_BASE_LIBRARY})
		else()
		set_target_properties(ggml-base PROPERTIES IMPORTED_LOCATION ${GGML_BASE_LIBRARY})
		endif()
		message(STATUS "Found ggml libraries: ${GGML_LIBRARY} ${GGML_BASE_LIBRARY}")
	endif()
	
	if(NOT DEFINED GGML_BACKEND_DL)
		include(CheckIncludeFileCXX)
		check_include_file_cxx(ggml-backend.h JJML_BACKEND_INCLUDE_FOUND)
		if(JJML_BACKEND_INCLUDE_FOUND)
			set(GGML_BACKEND_DL ON)
			message(STATUS "Found ggml-backend.h, enabling GGML_BACKEND_DL")
		endif()
	endif()
endif()

if(EXISTS "${llama_LIBRARY}") # llama.cpp as external library
	if(NOT llama_FOUND) # not CMake configured
		add_library(llama SHARED IMPORTED GLOBAL)
		if(MSVC)
		set_target_properties(llama PROPERTIES IMPORTED_IMPLIB ${llama_LIBRARY})
		else()
		set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${llama_LIBRARY})
		endif()
		target_link_libraries(llama INTERFACE ${GGML_LIBRARY})
		message(STATUS "Found LLAMA library: ${llama_LIBRARY}") 
	endif()
endif()

if(NOT JJML_DO_NOT_BUILD_TP)
	add_subdirectory(tp) # build locally if needed
endif()

message(STATUS "GGML_BUILD_NUMBER=${GGML_BUILD_NUMBER} LLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}")

#
# JJML NATIVE BUILD
#
if(GGML_BACKEND_DL)
	message(STATUS "Build JNI libraries with support for dynamic ggml backends") 
endif()

# Force C++17 and ISO C++
set(CMAKE_CXX_STANDARD 17)
if (NOT MSVC)
    add_compile_options(-pedantic-errors)
endif()

add_subdirectory(org_argeo_jjml_ggml)
add_subdirectory(org_argeo_jjml_llm)
                      