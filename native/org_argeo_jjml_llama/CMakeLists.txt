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
target_link_libraries(${TARGET} PRIVATE	ggml llama)
	
a2_jni_target(${TARGET})
