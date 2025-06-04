#
# THIRD PARTIES
#
add_subdirectory(tp)

if(NOT CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# JJML NATIVE BUILD
#
add_subdirectory(org_argeo_jjml_ggml)
add_subdirectory(org_argeo_jjml_llama)
