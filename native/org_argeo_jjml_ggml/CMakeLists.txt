set(TARGET Java_org_argeo_jjml_ggml)

add_library(${TARGET} SHARED 
	jjml_ggml_backend.cpp
)
target_link_libraries(${TARGET} PRIVATE	ggml)
	
a2_jni_target(${TARGET})
