#include <llama.h>

#include <cassert>

#include "argeo_jni_utils.h"

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppContext.h"
#include "org_argeo_jjml_llama_LlamaCppNative.h"

/*
 * PARAMETERS
 */
/** @brief Set context parameters from native to Java.*/
static void set_context_params(JNIEnv *env, jobject contextParams,
		llama_context_params ctx_params) {
	// integers
	env->SetIntField(contextParams, LlamaCppContextParams$contextSize,
			ctx_params.n_ctx);
	env->SetIntField(contextParams, LlamaCppContextParams$maxBatchSize,
			ctx_params.n_batch);
	env->SetIntField(contextParams, LlamaCppContextParams$physicalMaxBatchSize,
			ctx_params.n_ubatch);
	env->SetIntField(contextParams, LlamaCppContextParams$maxSequencesCount,
			ctx_params.n_seq_max);
	env->SetIntField(contextParams, LlamaCppContextParams$generationThreadCount,
			ctx_params.n_threads);
	env->SetIntField(contextParams, LlamaCppContextParams$batchThreadCount,
			ctx_params.n_threads_batch);

	// enums
	env->SetIntField(contextParams, LlamaCppContextParams$poolingTypeCode,
			ctx_params.pooling_type);

	// booleans
	env->SetBooleanField(contextParams, LlamaCppContextParams$embeddings,
			ctx_params.embeddings);
}

/** @brief Get context parameters from Java to native.*/
static void get_context_params(JNIEnv *env, jobject contextParams,
		llama_context_params *ctx_params) {
	// integers
	ctx_params->n_ctx = env->GetIntField(contextParams,
			LlamaCppContextParams$contextSize);
	ctx_params->n_batch = env->GetIntField(contextParams,
			LlamaCppContextParams$maxBatchSize);
	ctx_params->n_ubatch = env->GetIntField(contextParams,
			LlamaCppContextParams$physicalMaxBatchSize);
	ctx_params->n_seq_max = env->GetIntField(contextParams,
			LlamaCppContextParams$maxSequencesCount);
	ctx_params->n_threads = env->GetIntField(contextParams,
			LlamaCppContextParams$generationThreadCount);
	ctx_params->n_threads_batch = env->GetIntField(contextParams,
			LlamaCppContextParams$batchThreadCount);

	// enums
	switch (env->GetIntField(contextParams,
			LlamaCppContextParams$poolingTypeCode)) {
	case LLAMA_POOLING_TYPE_UNSPECIFIED:
		ctx_params->pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
		break;
	case LLAMA_POOLING_TYPE_NONE:
		ctx_params->pooling_type = LLAMA_POOLING_TYPE_NONE;
		break;
	case LLAMA_POOLING_TYPE_MEAN:
		ctx_params->pooling_type = LLAMA_POOLING_TYPE_MEAN;
		break;
	case LLAMA_POOLING_TYPE_CLS:
		ctx_params->pooling_type = LLAMA_POOLING_TYPE_CLS;
		break;
	case LLAMA_POOLING_TYPE_LAST:
		ctx_params->pooling_type = LLAMA_POOLING_TYPE_LAST;
		break;
	default:
		assert(!"Invalid pooling type value");
		break;
	}

	// booleans
	ctx_params->embeddings = env->GetBooleanField(contextParams,
			LlamaCppContextParams$embeddings);
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llama_LlamaCppNative_newContextParams(
		JNIEnv *env, jclass) {
	jobject res = env->NewObject(LlamaCppContextParams,
			LlamaCppContextParams$LlamaCppContextParams);
	llama_context_params ctx_params = llama_context_default_params();
	set_context_params(env, res, ctx_params);
	return res;
}

/*
 * LIFECYCLE
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doInit(
		JNIEnv *env, jobject obj, jobject modelObj, jobject contextParams) {
	llama_model *model = (llama_model*) getPointer(env, modelObj);

	llama_context_params ctx_params = llama_context_default_params();
	get_context_params(env, contextParams, &ctx_params);

	// Embedding config
	// FIXME make it configurable
//	ctx_params.embeddings = true;
//	ctx_params.n_batch = 2048;
//	ctx_params.n_ubatch = ctx_params.n_batch;

	llama_context *ctx = llama_new_context_with_model(model, ctx_params);

	if (ctx == NULL) {
		env->ThrowNew(IllegalStateException,
				"Failed to create llama.cpp context");
		return 0;
	}
	return (jlong) ctx;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doDestroy(
		JNIEnv *env, jobject obj) {
	llama_context *ctx = (llama_context*) getPointer(env, obj);
	llama_free(ctx);
}

/*
 * ACESSORS
 */
JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doGetPoolingType(
		JNIEnv *env, jobject obj) {
	llama_context *ctx = (llama_context*) getPointer(env, obj);
	return llama_pooling_type(ctx);
}

