#include <iostream>
#include <vector>
#include <algorithm>

#include <llama.h>

#include "org_argeo_jjml_llama_LlamaCppContext.h"
#include "org_argeo_jjml_llama_LlamaCppModel.h"
#include "org_argeo_jjml_llama_LlamaCppBackend.h"

/**
 * Utilities
 */
jlong getPointer(JNIEnv *env, jobject obj) {
	jclass clss = env->GetObjectClass(obj);
	jmethodID method = env->GetMethodID(clss, "getPointer", "()J");
	// TODO check that method was found
	jlong pointer = env->CallLongMethod(obj, method);
	return pointer;
}

/*
 * Llama context
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doInit(
		JNIEnv *env, jobject obj, jobject modelObj) {
	llama_model *model = (llama_model*) getPointer(env, modelObj);

	llama_context_params ctx_params = llama_context_default_params();

	llama_context *ctx = llama_new_context_with_model(model, ctx_params);

	if (ctx == NULL) {
		fprintf(stderr, "%s: error: failed to create the llama_context\n",
				__func__);
		jclass IllegalStateException = env->FindClass(
				"java/lang/IllegalStateException");
		env->ThrowNew(IllegalStateException, "Failed to create llama context");
	}
	return (jlong) ctx;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doDestroy(
		JNIEnv *env, jobject obj) {
	llama_context *ctx = (llama_context*) getPointer(env, obj);
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
		JNIEnv *env, jobject obj) {
	llama_model *model = (llama_model*) getPointer(env, obj);
	llama_free_model(model);
}

JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doTokenize(
		JNIEnv *env, jobject obj, jstring str, jboolean add_special,
		jboolean parse_special) {
	llama_model *model = (llama_model*) getPointer(env, obj);

	const char *chars = env->GetStringUTFChars(str, NULL);
	std::string text(chars);
	int text_length = env->GetStringLength(str);

	// upper limit for the number of tokens
	int n_tokens = text_length + 2 * add_special;
	std::vector<llama_token> result(n_tokens);
	n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(),
			result.size(), add_special, parse_special);
	if (n_tokens < 0) {
		result.resize(-n_tokens);
		int check = llama_tokenize(model, text.data(), text.length(),
				result.data(), result.size(), add_special, parse_special);
		GGML_ASSERT(check == -n_tokens);
	} else {
		result.resize(n_tokens);
	}
	jintArray res = env->NewIntArray(result.size());
	env->SetIntArrayRegion(res, 0, result.size(), result.data());
	return res;
}

JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDeTokenize(
		JNIEnv *env, jobject obj, jintArray tokens, jboolean special) {
	llama_model *model = (llama_model*) getPointer(env, obj);

	jsize tokens_size = env->GetArrayLength(tokens);
	jboolean *isCopy;
	void *region = env->GetPrimitiveArrayCritical(tokens, isCopy);

	std::string text;
	//text.resize(std::max(text.capacity(), tokens_size));
	int32_t n_chars = llama_detokenize(model, (int*) region, tokens_size,
			&text[0], (int32_t) text.size(), false, special);
	if (n_chars < 0) {
		text.resize(-n_chars);
		n_chars = llama_detokenize(model, (int*) region, tokens_size, &text[0],
				(int32_t) text.size(), false, special);
		GGML_ASSERT(n_chars <= (int32_t )text.size()); // whitespace trimming is performed after per-token detokenization
	}
	env->ReleasePrimitiveArrayCritical(tokens, region, 0);

	text.resize(n_chars);
	jstring res = env->NewStringUTF(text.data());
	return res;
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

