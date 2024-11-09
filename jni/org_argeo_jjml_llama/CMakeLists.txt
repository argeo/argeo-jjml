set(TARGET org_argeo_jjml_llama)

add_library(${TARGET} SHARED 
	org_argeo_jjml_llama_.cpp
	jjml_llama.cpp
	jjml_llama_vocabulary.cpp
	jjml_llama_model.cpp
	jjml_llama_context.cpp
	jjml_llama_sampling.cpp
	jjml_llama_batch.cpp
	jjml_llama_embedding.cpp
)
target_include_directories(${TARGET} PRIVATE ${JNI_INCLUDE_DIRS} ../include org_argeo_jjml_llama )
target_link_libraries(${TARGET} PRIVATE ggml llama)

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set_target_properties(${TARGET} PROPERTIES 
	LIBRARY_OUTPUT_DIRECTORY
	"${CMAKE_BINARY_DIR}/../a2/${A2_TARGET_ARCH_CATEGORY_PREFIX}/${A2_CATEGORY}"
)

# MSYS2
#link_directories("/ucrt64/lib/")
#target_link_libraries(${TARGET} PRIVATE libggml libllama)

install(TARGETS ${TARGET} LIBRARY)
