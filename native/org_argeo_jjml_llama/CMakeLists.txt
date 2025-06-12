set(TARGET Java_org_argeo_jjml_llama)

add_library(${TARGET} SHARED 
	org_argeo_jjml_llama_.cpp
	jjml_llama.cpp
	jjml_llama_vocabulary.cpp
	jjml_llama_model.cpp
	jjml_llama_context.cpp
	jjml_llama_sampling.cpp
	jjml_llama_batch_write.cpp
	jjml_llama_batch_read.cpp
	jjml_llama_embedding.cpp
)

if(ggml_FOUND)
target_link_libraries(${TARGET} PRIVATE	ggml::ggml ggml::ggml-base llama)
else()
target_link_libraries(${TARGET} PRIVATE	ggml ggml-base llama)
endif()

#target_compile_definitions(${TARGET} PRIVATE GGML_ASSERT=)

target_compile_definitions(${TARGET} PRIVATE
_JJML_PRE_LLAMA_0_0_4881
)

a2_jni_target(${TARGET})
