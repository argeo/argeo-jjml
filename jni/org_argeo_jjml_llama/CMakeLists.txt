set(TARGET org_argeo_jjml_llama)

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
target_include_directories(${TARGET} PRIVATE 
	../include/org_argeo_jjml ../include ${JNI_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE
	ggml llama)

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
	"${A2_BASE}/${A2_TARGET_ARCH_CATEGORY_PREFIX}/${A2_CATEGORY}"
)

install(TARGETS ${TARGET} LIBRARY)
