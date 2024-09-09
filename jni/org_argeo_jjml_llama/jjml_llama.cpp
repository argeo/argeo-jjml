#include <llama.h>
#include <stddef.h>

#include "org_argeo_jjml_llama_LlamaCppContext.h"
#include "org_argeo_jjml_llama_LlamaCppModel.h"
#include "org_argeo_jjml_llama_LlamaCppBackend.h"

/*
 * Lllama context
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doInit(
		JNIEnv *env, jobject, jlong modelPointer) {
	llama_model *model = (struct llama_model*) modelPointer;

	llama_context_params ctx_params = llama_context_default_params();

	llama_context *ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        jclass IllegalStateException = env->FindClass("java/lang/IllegalStateException");
        env->ThrowNew(IllegalStateException, "Failed to create llama context");
    }
	return (jlong) ctx;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doDestroy(
		JNIEnv*, jobject, jlong pointer) {
	llama_context *ctx = (llama_context*) pointer;
	llama_free(ctx);
}

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
	llama_model *model = (llama_model*) pointer;
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
