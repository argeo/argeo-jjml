#
# THIRD PARTIES
#
add_subdirectory(tp)

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
message (STATUS "Will build JNI libraries with support for dynamic ggml backends") 
endif()

add_subdirectory(org_argeo_jjml_ggml)
add_subdirectory(org_argeo_jjml_llama)
