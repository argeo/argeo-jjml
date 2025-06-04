set(TARGET Java_org_argeo_jjml_ggml)

add_library(${TARGET} SHARED 
	jjml_ggml_backend.cpp
)
target_link_libraries(${TARGET} PRIVATE	ggml)
if (GGML_BACKEND_DL)
    target_compile_definitions(${TARGET} PRIVATE GGML_BACKEND_DL)
endif()
	
a2_jni_target(${TARGET})
