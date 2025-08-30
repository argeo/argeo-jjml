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

# When breaking changes are not funcitonnaly critical
# we keep building against older, in order to extend the life
# of a given JJML branch. To be trimmed when a new branch
# is created.
if(LLAMA_BUILD_NUMBER LESS 5913)
target_compile_definitions(${TARGET} PRIVATE JJML_PRE_LLAMA_0_0_5913)
endif()
if(LLAMA_BUILD_NUMBER LESS 5882)
target_compile_definitions(${TARGET} PRIVATE JJML_PRE_LLAMA_0_0_5882)
endif()
if(LLAMA_BUILD_NUMBER LESS 4881)
target_compile_definitions(${TARGET} PRIVATE JJML_PRE_LLAMA_0_0_4881)
endif()

a2_jni_target(${TARGET})
