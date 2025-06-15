set(TARGET Java_org_argeo_jjml_llm)

add_library(${TARGET} SHARED 
	org_argeo_jjml_llm_.cpp
	jjml_llm.cpp
	jjml_llm_vocabulary.cpp
	jjml_llm_model.cpp
	jjml_llm_chat.cpp
	jjml_llm_context.cpp
	jjml_llm_sampling.cpp
	jjml_llm_batch_write.cpp
	jjml_llm_batch_read.cpp
	jjml_llm_embedding.cpp
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
