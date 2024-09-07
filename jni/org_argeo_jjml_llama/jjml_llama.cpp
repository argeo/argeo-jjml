#include "org_argeo_jjml_llama_LlamaCppBackend.h"
#include "org_argeo_jjml_llama_LlamaCppModel.h"

#include "llama.h"

/*
 * Llama model
 */
JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doInit(JNIEnv*,
		jobject) {
	//llama_load_model_from_file(path_model, params);
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDestroy(
		JNIEnv*, jobject) {

}

/*
 * Llama backend
 */
JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_doInit(JNIEnv*,
		jobject) {
	llama_backend_init();
	//llama_numa_init(params.numa);
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_doDestroy(
		JNIEnv*, jobject) {
	llama_backend_free();
}
