#include <argeo/argeo_jni.h>
#include <jni.h>
#include <jni_md.h>
#include <llama.h>
#include <stddef.h>
#include <cassert>

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppContext.h" // IWYU pragma: keep
#include "org_argeo_jjml_llama_LlamaCppNative.h" // IWYU pragma: keep

/*
 * PARAMETERS
 */
/** @brief Set context parameters from native to Java.*/
//static void set_context_params(JNIEnv *env, jobject contextParams,
//		llama_context_params ctx_params) {
//	// integers
//	env->SetIntField(contextParams, LlamaCppContextParams$contextSize,
//			ctx_params.n_ctx);
//	env->SetIntField(contextParams, LlamaCppContextParams$maxBatchSize,
//			ctx_params.n_batch);
//	env->SetIntField(contextParams, LlamaCppContextParams$physicalMaxBatchSize,
//			ctx_params.n_ubatch);
//	env->SetIntField(contextParams, LlamaCppContextParams$maxSequencesCount,
//			ctx_params.n_seq_max);
//	env->SetIntField(contextParams, LlamaCppContextParams$generationThreadCount,
//			ctx_params.n_threads);
//	env->SetIntField(contextParams, LlamaCppContextParams$batchThreadCount,
//			ctx_params.n_threads_batch);
//
//	// enums
//	env->SetIntField(contextParams, LlamaCppContextParams$poolingTypeCode,
//			ctx_params.pooling_type);
//
//	// booleans
//	env->SetBooleanField(contextParams, LlamaCppContextParams$embeddings,
//			ctx_params.embeddings);
//}
/** @brief Get context parameters from Java to native.*/
static void get_context_params(JNIEnv *env, jobject params,
		llama_context_params *ctx_params) {
	jclass clss = env->FindClass((JNI_PKG + "LlamaCppContext$Params").c_str());
	// integers
	ctx_params->n_ctx = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_ctx", "()I"));
	ctx_params->n_batch = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_batch", "()I"));
	ctx_params->n_ubatch = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_ubatch", "()I"));
	ctx_params->n_seq_max = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_seq_max", "()I"));
	ctx_params->n_threads = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_threads", "()I"));
	ctx_params->n_threads_batch = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_threads_batch", "()I"));
//	ctx_params->n_batch = env->GetIntField(params,
//			LlamaCppContextParams$maxBatchSize);
//	ctx_params->n_ubatch = env->GetIntField(params,
//			LlamaCppContextParams$physicalMaxBatchSize);
//	ctx_params->n_seq_max = env->GetIntField(params,
//			LlamaCppContextParams$maxSequencesCount);
//	ctx_params->n_threads = env->GetIntField(params,
//			LlamaCppContextParams$generationThreadCount);
//	ctx_params->n_threads_batch = env->GetIntField(params,
//			LlamaCppContextParams$batchThreadCount);

// enums
	switch (env->CallIntMethod(params,
			env->GetMethodID(clss, "pooling_type", "()I"))) {
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
	ctx_params->embeddings = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "embeddings", "()Z"));
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llama_LlamaCppNative_newContextParams(
		JNIEnv *env, jclass) {
	jclass clss = env->FindClass((JNI_PKG + "LlamaCppContext$Params").c_str());
	// Tip: in order to find the constructor signature, use:
	// javap -s '../org.argeo.jjml/bin/org/argeo/jjml/llama/LlamaCppContext$Params.class'
	jmethodID constructor = env->GetMethodID(clss, "<init>",
			"(IIIIIIIIIFFFFFFIFIIZZZZ)V");
	llama_context_params ctx_params = llama_context_default_params();
	jobject res = env->NewObject(clss, constructor, //
			ctx_params.n_ctx, //
			ctx_params.n_batch, //
			ctx_params.n_ubatch, //
			ctx_params.n_seq_max, //
			ctx_params.n_threads, //
			ctx_params.n_threads_batch, //
			ctx_params.rope_scaling_type, //
			ctx_params.pooling_type, //
			ctx_params.attention_type, //
			ctx_params.rope_freq_base, //
			ctx_params.rope_freq_scale, //
			ctx_params.yarn_ext_factor, //
			ctx_params.yarn_attn_factor, //
			ctx_params.yarn_beta_fast, //
			ctx_params.yarn_beta_slow, //
			ctx_params.yarn_orig_ctx, //
			ctx_params.defrag_thold, //
			ctx_params.type_k, //
			ctx_params.type_v, //
			ctx_params.embeddings, //
			ctx_params.offload_kqv, //
			ctx_params.flash_attn, //
			ctx_params.no_perf //
			);
//	set_context_params(env, res, ctx_params);
	return res;
}

/*
 * LIFECYCLE
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doInit(
		JNIEnv *env, jobject, jlong modelPointer, jobject contextParams) {
	//auto *model = getPointer<llama_model*>(env, modelObj);
	auto *model = argeo::jni::getPointer<llama_model*>(modelPointer);

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
		JNIEnv *env, jobject obj, jlong pointer) {
	auto *ctx = argeo::jni::getPointer<llama_context*>(pointer);
	llama_free(ctx);
}

/*
 * ACESSORS
 */
JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doGetPoolingType(
		JNIEnv *env, jobject obj, jlong pointer) {
	auto *ctx = argeo::jni::getPointer<llama_context*>(pointer);
	return llama_pooling_type(ctx);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doGetContextSize(
		JNIEnv*, jobject, jlong pointer) {
	auto *ctx = argeo::jni::getPointer<llama_context*>(pointer);
	return llama_n_ctx(ctx);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doGetBatchSize(
		JNIEnv*, jobject, jlong pointer) {
	auto *ctx = argeo::jni::getPointer<llama_context*>(pointer);
	return llama_n_batch(ctx);
}
