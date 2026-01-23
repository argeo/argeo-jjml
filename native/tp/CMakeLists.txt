#
# BUILD THIRD PARTY FROM SUBMODULES
#

if(GGML_BACKEND_DL)
set(BUILD_SHARED_LIBS ON)
endif()

set(TARGET_NATIVE_OUTPUT_GGML ${A2_OUTPUT}/lib/${TARGET_NATIVE_CATEGORY_PREFIX}/org.argeo.tp.ggml)
set(TARGET_NATIVE_OUTPUT_LLAMA ${TARGET_NATIVE_OUTPUT_GGML})

# All CPU variants backends take the default output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_GGML}>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_GGML}>)

if(MINGW)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_GGML}>)
elseif(MSVC)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_GGML}>)
else()
endif()

if(NOT EXISTS ${GGML_LIBRARY})
	## multiple backends libraries
	if(NOT DEFINED GGML_BACKEND_DL AND NOT DEFINED GGML_NATIVE AND NOT DEFINED GGML_CPU_ALL_VARIANTS)
		set(GGML_BACKEND_DL ON)
		set(GGML_CPU_ALL_VARIANTS ON)
		set(GGML_NATIVE OFF)
		message(STATUS "Enabling GGML_BACKEND_DL with all CPU variants by default")
	endif()

	# build locally
	if(JJML_FORCE_BUILD_LLAMA_GGML)
	add_subdirectory(llama.cpp/ggml)
	else() # default
	add_subdirectory(ggml)
	endif()
	message(STATUS "Build ggml locally from submodule native/tp/ggml") 
endif()


if(NOT EXISTS "${llama_LIBRARY}")
	set(LLAMA_STANDALONE ON)
	add_subdirectory(llama.cpp) # build locally
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
			set_target_properties(mtmd PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_LLAMA})
		endif()
	endif()
	message(STATUS "Build llama.cpp locally from submodule native/tp/llama.cpp") 
endif()

set(TARGET_NATIVE_OUTPUT_WHISPER ${TARGET_NATIVE_OUTPUT_GGML})
if(NOT EXISTS "${whisper_LIBRARY}")
	# workaround for disappearance of GGML_KQ_MASK_PAD
	add_compile_definitions(GGML_KQ_MASK_PAD=64)
	
	add_subdirectory(whisper.cpp) # build locally
	if(MINGW)
		set_target_properties(whisper PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_WHISPER})
	elseif(MSVC)
		set_target_properties(whisper PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_WHISPER}>)
	else()
		set_target_properties(whisper PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TARGET_NATIVE_OUTPUT_WHISPER})
	endif()
	if(WHISPER_BUILD_EXAMPLES)
		set_target_properties(whisper-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${TARGET_NATIVE_OUTPUT_WHISPER}>)
	endif()
	message(STATUS "Build whisper.cpp locally from submodule native/tp/whisper.cpp") 
endif()

# ggml and llama.cpp versions

# hack to retrieve GGML base version
# see https://stackoverflow.com/a/15519607
file(STRINGS ggml/CMakeLists.txt VersionSetter
     REGEX "^[ ]*set\\(GGML_VERSION_MAJOR")
string(REGEX REPLACE "(^[ ]*set\\(GGML_VERSION_MAJOR[ ]*)([^\\)]*)\\)" "\\2"
       GGML_VERSION_MAJOR ${VersionSetter})
file(STRINGS ggml/CMakeLists.txt VersionSetter
     REGEX "^[ ]*set\\(GGML_VERSION_MINOR")
string(REGEX REPLACE "(^[ ]*set\\(GGML_VERSION_MINOR[ ]*)([^\\)]*)\\)" "\\2"
       GGML_VERSION_MINOR ${VersionSetter})
file(STRINGS ggml/CMakeLists.txt VersionSetter
     REGEX "^[ ]*set\\(GGML_VERSION_PATCH")
string(REGEX REPLACE "(^[ ]*set\\(GGML_VERSION_PATCH[ ]*)([^\\)]*)\\)" "\\2"
       GGML_VERSION_PATCH ${VersionSetter})

# hack to retrieve whisper.cpp base version
file(STRINGS whisper.cpp/CMakeLists.txt VersionSetter
     REGEX "[ \t]*project\\(\"whisper.cpp\" VERSION")
string(REGEX MATCH "VERSION[ ]+([^ )]+)" version_param "${VersionSetter}")
if(version_param)
    set(WHISPER_CMAKE_VERSION "${CMAKE_MATCH_1}")
endif()
message(STATUS "WHISPER_CMAKE_VERSION=${WHISPER_CMAKE_VERSION}")


if (Git_FOUND)
execute_process(COMMAND "${GIT_EXECUTABLE}" describe --exact-match --tags
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ggml
 OUTPUT_VARIABLE JJML_GGML_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT JJML_GGML_TAG)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-list --count v${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}..HEAD
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ggml
 OUTPUT_VARIABLE JJML_GGML_COMMIT_COUNT OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-list --count HEAD
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp
 OUTPUT_VARIABLE JJML_LLAMA_COMMIT_COUNT OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" describe --exact-match --tags
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp
 OUTPUT_VARIABLE JJML_WHISPER_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT JJML_WHISPER_TAG)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-list --count v${WHISPER_CMAKE_VERSION}..HEAD
 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp
 OUTPUT_VARIABLE JJML_WHISPER_COMMIT_COUNT OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT GGML_BUILD_NUMBER)
set(GGML_BUILD_NUMBER ${JJML_GGML_COMMIT_COUNT} CACHE INTERNAL GGML_BUILD_NUMBER)
endif()
if(NOT GGML_VERSION)
if (JJML_GGML_COMMIT_COUNT)
set(GGML_VERSION ${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}.${JJML_GGML_COMMIT_COUNT} CACHE INTERNAL GGML_VERSION)
else()
set(GGML_VERSION ${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH} CACHE INTERNAL GGML_VERSION)
endif()
endif()

if(NOT LLAMA_BUILD_NUMBER)
set(LLAMA_BUILD_NUMBER ${JJML_LLAMA_COMMIT_COUNT} CACHE INTERNAL LLAMA_BUILD_NUMBER)
endif()
if(NOT LLAMA_VERSION)
set(LLAMA_VERSION 0.0.${JJML_LLAMA_COMMIT_COUNT} CACHE INTERNAL LLAMA_VERSION)
endif()

if(NOT WHISPER_VERSION)
if (JJML_WHISPER_COMMIT_COUNT)
set(WHISPER_VERSION ${WHISPER_CMAKE_VERSION}.${JJML_WHISPER_COMMIT_COUNT} CACHE INTERNAL WHISPER_VERSION)
else()
set(WHISPER_VERSION ${WHISPER_CMAKE_VERSION} CACHE INTERNAL WHISPER_VERSION)
endif()
endif()

file(WRITE versions.mk "GGML_VERSION=${GGML_VERSION}\n")
file(APPEND versions.mk "LLAMA_VERSION=${LLAMA_VERSION}\n")
file(APPEND versions.mk "WHISPER_VERSION=${WHISPER_VERSION}\n")
