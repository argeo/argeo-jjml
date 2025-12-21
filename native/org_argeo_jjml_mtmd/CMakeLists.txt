set(TARGET Java_org_argeo_jjml_mtmd)

add_library(${TARGET} SHARED 
	jjml_mtmd.cpp
	jjml_mtmd_utils.cpp
)

if(ggml_FOUND)
target_link_libraries(${TARGET} PRIVATE	ggml::ggml ggml::ggml-base llama mtmd)
else()
target_link_libraries(${TARGET} PRIVATE	ggml ggml-base llama mtmd)
endif()

a2_jni_target(${TARGET})
