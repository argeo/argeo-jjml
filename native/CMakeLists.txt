#
# OPTIONS
#
option(JJML_FORCE_BUILD_TP "Force build of reference third-party submodules even if they are available on the system")
option(JJML_DO_NOT_BUILD_TP "Fail if third-party libraries are not found, instead of building them from the reference submodules")
option(JJML_JDK_TP "Use third-party environment provided by a ggml-instrumented JDK (useful for Windows)")
option(JJML_FORCE_LLAMA_BUILD_NUMBER "Force LLAMA_BUILD_NUMBER, in environments where it is not autoamtically set.")

#
# THIRD PARTY DETECTION
#
#set(CMAKE_FIND_DEBUG_MODE ON)

if(NOT JJML_FORCE_BUILD_TP)

if(NOT MSVC)
# Debian official until libraries are stable
set(CMAKE_PREFIX_PATH
/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/llama/cmake-private
)
endif()

if(JJML_JDK_TP) # Use environment provided by a ggml-instrumented JDK
	set(CMAKE_LIBRARY_PATH ${JAVA_HOME}/bin ${JAVA_HOME}/lib)
	include_directories(${JAVA_HOME}/include SYSTEM)
endif()

# look for llama.cpp via CMake configs (preferred)
find_package(llama QUIET)
find_package(whisper QUIET)

if(whisper_FOUND)
	message(STATUS "Found whisper packages: ${whisper_LIBRARY} ${GGML_LIBRARY} ${GGML_BASE_LIBRARY}")
endif()

if(llama_FOUND)
	# Workaround for mtmd not being properly configured in CMake
	find_library(mtmd_LIBRARY mtmd HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/llama)

	message(STATUS "Found llama.cpp and ggml packages: ${llama_LIBRARY} ${mtmd_LIBRARY} ${GGML_LIBRARY} ${GGML_BASE_LIBRARY}")
	
	if(NOT whisper_FOUND)
		find_library(whisper_LIBRARY whisper)
	endif()
else()
	# look for llama.cpp via simple library search
	find_library(llama_LIBRARY llama)
	if(NOT whisper_FOUND)
		find_library(whisper_LIBRARY whisper)
	endif()
	
	# look for ggml via CMake configs (preferred)
	find_package(ggml QUIET)
	if(ggml_FOUND)
		set(LLAMA_USE_SYSTEM_GGML ON)
		
		# override default
		set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
		set(WHISPER_USE_SYSTEM_GGML ON)
		
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
		if(WIN32)
		set_target_properties(ggml PROPERTIES IMPORTED_IMPLIB ${GGML_LIBRARY})
		else()
		set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${GGML_LIBRARY})
		endif()
		
		add_library(ggml-base SHARED IMPORTED GLOBAL)
		if(WIN32)
		set_target_properties(ggml-base PROPERTIES IMPORTED_IMPLIB ${GGML_BASE_LIBRARY})
		else()
		set_target_properties(ggml-base PROPERTIES IMPORTED_LOCATION ${GGML_BASE_LIBRARY})
		endif()
		message(STATUS "Found ggml libraries: ${GGML_LIBRARY} ${GGML_BASE_LIBRARY}")
	endif()
	
	if(NOT DEFINED GGML_BACKEND_DL)
			message(STATUS "## CHECK ###")
		include(CheckIncludeFileCXX)
		check_include_file_cxx(ggml-backend.h JJML_BACKEND_INCLUDE_FOUND)
		if(JJML_BACKEND_INCLUDE_FOUND)
			set(GGML_BACKEND_DL ON)
			message(STATUS "Found ggml-backend.h, enabling GGML_BACKEND_DL")
		endif()
	endif()
endif()

if(EXISTS "${llama_LIBRARY}") # libllama as external library
	if(NOT llama_FOUND) # not CMake configured
		add_library(llama SHARED IMPORTED GLOBAL)
		if(WIN32)
		set_target_properties(llama PROPERTIES IMPORTED_IMPLIB ${llama_LIBRARY})
		else()
		set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${llama_LIBRARY})
		endif()
		target_link_libraries(llama INTERFACE ${GGML_LIBRARY})
		message(STATUS "Found llama.cpp library: ${llama_LIBRARY}") 
	endif()
endif()

# Workaround for mtmd not being properly configured in CMake
if(EXISTS "${mtmd_LIBRARY}") # mtmd as external library
	if(NOT mtmd_FOUND) # not CMake configured
		add_library(mtmd SHARED IMPORTED GLOBAL)
		if(WIN32)
		set_target_properties(mtmd PROPERTIES IMPORTED_IMPLIB ${mtmd_LIBRARY})
		else()
		set_target_properties(mtmd PROPERTIES IMPORTED_LOCATION ${mtmd_LIBRARY})
		endif()
		target_link_libraries(mtmd INTERFACE ${llama_LIBRARY} ${GGML_LIBRARY})
		message(STATUS "Found mtmd library: ${mtmd_LIBRARY}") 
	endif()
endif()

if(EXISTS "${whisper_LIBRARY}") # whisper as external library
	if(NOT whisper_FOUND) # not CMake configured
		add_library(whisper SHARED IMPORTED GLOBAL)
		if(WIN32)
		set_target_properties(whisper PROPERTIES IMPORTED_IMPLIB ${whisper_LIBRARY})
		else()
		set_target_properties(whisper PROPERTIES IMPORTED_LOCATION ${whisper_LIBRARY})
		endif()
		target_link_libraries(whisper INTERFACE ${GGML_LIBRARY})
		message(STATUS "Found whisper library: ${mtmd_LIBRARY}") 
	endif()
endif()

if(EXISTS "${llama_LIBRARY}" AND EXISTS "${mtmd_LIBRARY}" AND EXISTS "${whisper_LIBRARY}")
	message(STATUS "Build will only use system libraries.")
else()
	if(NOT JJML_DO_NOT_BUILD_TP)
		# Build reference third-party components if needed
		add_subdirectory(tp)
	else()
		message(FATAL "Not all system libraries where found and build of third-party components is disabled.")
	endif()
endif()

# FIXME - LLAMA_BUILD_NUMBER is set to 1 in the MSYS-provided libraries
# It should be reported to these packagers.
if(JJML_FORCE_LLAMA_BUILD_NUMBER)
	set(LLAMA_BUILD_NUMBER ${JJML_FORCE_LLAMA_BUILD_NUMBER})
	set(LLAMA_VERSION 0.0.${LLAMA_BUILD_NUMBER})
else()
	if(LLAMA_BUILD_NUMBER EQUAL 1)
	message(WARNING "LLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER} is wrong, force it with JJML_FORCE_LLAMA_BUILD_NUMBER")
	endif()
endif()

message(STATUS "GGML_BUILD_NUMBER=${GGML_BUILD_NUMBER} LLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}")
message(STATUS "GGML_VERSION=${GGML_VERSION} LLAMA_VERSION=${LLAMA_VERSION} WHISPER_VERSION=${WHISPER_VERSION}")

#
# JJML NATIVE BUILD
#
if(GGML_BACKEND_DL)
	message(STATUS "Build JNI libraries with support for dynamic ggml backends") 
endif()

# Force ISO C++
if (MSVC)
add_compile_options(/permissive-)
else()
add_compile_options(-pedantic-errors)
endif()

# JNI shared libraries
add_subdirectory(org_argeo_jjml_ggml)
add_subdirectory(org_argeo_jjml_llm)

add_subdirectory(org_argeo_jjml_mtmd)

add_subdirectory(org_argeo_jjml_whisper)

                     