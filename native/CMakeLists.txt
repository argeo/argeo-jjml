#
# THIRD PARTIES
#

# Debug
#set(CMAKE_FIND_DEBUG_MODE ON)

# We always want the shared libraries
set(BUILD_SHARED_LIBS ON)

if(NOT MSVC)
# Debian official until libraries are stable
set(CMAKE_PREFIX_PATH
/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/ggml/cmake-private
/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/llama/cmake-private
)
endif()

set(TARGET_NATIVE_OUTPUT_GGML ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX}/org.argeo.tp.ggml)
set(TARGET_NATIVE_OUTPUT_LLAMA ${TARGET_NATIVE_OUTPUT_GGML})

# All CPU variants backends take the default
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_GGML}>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_GGML}>)

if(MINGW)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_GGML}>)
elseif(MSVC)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_GGML}>)
else()
endif()

message(STATUS "CMAKE_LIBRARY_ARCHITECTURE=${CMAKE_LIBRARY_ARCHITECTURE}")

if(NOT JJML_FORCE_BUILD_TP)
	if(JAVA_HOME) # Use environment provided by ggml-instrumented JDK
		set(CMAKE_LIBRARY_PATH ${JAVA_HOME}/bin ${JAVA_HOME}/lib)
		include_directories(${JAVA_HOME}/include SYSTEM)
	endif()

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
		if(MSVC)
		set_target_properties(llama PROPERTIES IMPORTED_IMPLIB ${llama_LIBRARY})
		else()
		set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${llama_LIBRARY})
		endif()
		target_link_libraries(llama INTERFACE ${GGML_LIBRARY})
	endif()
	message(STATUS "Found LLAMA library: ${llama_LIBRARY}") 
else()
	set(LLAMA_STANDALONE ON)
	add_subdirectory(tp/llama.cpp) # build locally
	if(MINGW)
		set_target_properties(llama PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_LLAMA})
		if(LLAMA_BUILD_TOOLS)
			set_target_properties(mtmd PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_LLAMA})
		endif()
	elseif(MSVC)
		set_target_properties(llama PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_LLAMA}>)
		if(LLAMA_BUILD_TOOLS)
			set_target_properties(mtmd PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_LLAMA}>)
		endif()
	else()
		set_target_properties(llama PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_LLAMA})
		if(LLAMA_BUILD_TOOLS)
			set_target_properties(mtmd PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_LLAMA})
		endif()
	endif()
	if(LLAMA_BUILD_TOOLS)
		set_target_properties(llama-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_LLAMA}>)
	endif()
	message(STATUS "Build llama.cpp locally from submodule native/tp/llama.cpp") 
endif()

# ggml and llama.cpp versions
if(NOT GGML_BUILD_NUMBER)
if (Git_FOUND)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-list --count HEAD
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tp/ggml
 OUTPUT_VARIABLE GGML_BUILD_NUMBER OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
endif()
if(NOT LLAMA_BUILD_NUMBER)
if (Git_FOUND)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-list --count HEAD
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tp/llama.cpp
 OUTPUT_VARIABLE LLAMA_BUILD_NUMBER OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
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
                      