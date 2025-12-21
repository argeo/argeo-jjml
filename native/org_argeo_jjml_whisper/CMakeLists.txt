set(TARGET Java_org_argeo_jjml_whisper)

add_library(${TARGET} SHARED 
	jjml_whisper.cpp
)

if(ggml_FOUND)
target_link_libraries(${TARGET} PRIVATE	ggml::ggml ggml::ggml-base whisper)
else()
target_link_libraries(${TARGET} PRIVATE	ggml ggml-base whisper)
endif()

a2_jni_target(${TARGET})
