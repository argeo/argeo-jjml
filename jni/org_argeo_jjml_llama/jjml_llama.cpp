#include "org_argeo_jjml_llama_LlamaCppBackend.h"
#include "org_argeo_jjml_llama_LlamaCppModel.h"

#include "llama.h"

/*
 * Llama model
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doInit(
		JNIEnv *env, jobject obj, jstring localPath) {
	const char *path_model = env->GetStringUTFChars(localPath, NULL);

	llama_model_params mparams = llama_model_default_params();

	llama_model *model = llama_load_model_from_file(path_model, mparams);

	env->ReleaseStringUTFChars(localPath, path_model);
	return (jlong) model;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDestroy(
		JNIEnv*, jobject, jlong pointer) {
	llama_model* model;
	model = (struct llama_model*) pointer;
	llama_free_model(model);
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
