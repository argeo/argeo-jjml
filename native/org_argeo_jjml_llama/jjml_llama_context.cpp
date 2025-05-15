#include <cassert>
#include <stdexcept>
#include <string>
#include <thread>
#include <iostream>

#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppBackend.h" // IWYU pragma: keep
#include "org_argeo_jjml_llama_LlamaCppContext.h" // IWYU pragma: keep

static struct ggml_threadpool *threadpool = NULL;

/*
 * PARAMETERS
 */
/** @brief Get context parameters from Java to native.*/
static void get_context_params(JNIEnv *env, jobject params,
		llama_context_params *ctx_params) {
	jclass clss = env->FindClass(JCLASS_CONTEXT_PARAMS.c_str());
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

	// TODO support more types
	int type_k = env->CallIntMethod(params,
			env->GetMethodID(clss, "type_k", "()I"));
	switch (env->CallIntMethod(params,
			env->GetMethodID(clss, "type_k", "()I"))) {
	case GGML_TYPE_F16:
		ctx_params->type_k = GGML_TYPE_F16;
		break;
	case GGML_TYPE_Q4_0:
		ctx_params->type_k = GGML_TYPE_Q4_0;
		break;
	case GGML_TYPE_Q8_0:
		ctx_params->type_k = GGML_TYPE_Q8_0;
		break;
	default:
		assert(!"Unsupported type_k type value");
		break;
	}

	switch (env->CallIntMethod(params,
			env->GetMethodID(clss, "type_v", "()I"))) {
	case GGML_TYPE_F16:
		ctx_params->type_v = GGML_TYPE_F16;
		break;
	case GGML_TYPE_Q4_0:
		ctx_params->type_v = GGML_TYPE_Q4_0;
		break;
	case GGML_TYPE_Q8_0:
		ctx_params->type_v = GGML_TYPE_Q8_0;
		break;
	default:
		assert(!"Unsupported type_k type value");
		break;
	}

	// booleans
	ctx_params->embeddings = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "embeddings", "()Z"));
	ctx_params->offload_kqv = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "offload_kqv", "()Z"));
	ctx_params->flash_attn = env->CallBooleanMethod(params,
			env->GetMethodID(clss, "flash_attn", "()Z"));
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_newContextParams(
		JNIEnv *env, jclass) {
	llama_context_params ctx_params = llama_context_default_params();
	jobject res = env->NewObject(
			argeo::jni::find_jclass(env, JCLASS_CONTEXT_PARAMS), //
			ContextParams$init, //
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
	return res;
}

/*
 * LIFECYCLE
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doInit(
		JNIEnv *env, jclass, jobject modelObj, jobject contextParams) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(env, modelObj);

		llama_context_params ctx_params = llama_context_default_params();
		get_context_params(env, contextParams, &ctx_params);

		llama_context *ctx = llama_init_from_model(model, ctx_params);
		if (ctx == NULL) {
			throw std::runtime_error("Failed to create llama.cpp context");
		}

		// Thread pool
		auto *reg = ggml_backend_dev_backend_reg(
				ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
		auto *ggml_threadpool_new_fn =
				(decltype(ggml_threadpool_new)*) ggml_backend_reg_get_proc_address(
						reg, "ggml_threadpool_new");
		auto *ggml_threadpool_free_fn =
				(decltype(ggml_threadpool_free)*) ggml_backend_reg_get_proc_address(
						reg, "ggml_threadpool_free");

	    unsigned int n_threads_os = std::thread::hardware_concurrency();
//		struct ggml_threadpool_params tpp_batch;
//	    ggml_threadpool_params_init(&tpp_batch, n_threads);
		struct ggml_threadpool_params tpp;
		ggml_threadpool_params_init(&tpp, n_threads_os);

		//set_process_priority(params.cpuparams.priority);

//		struct ggml_threadpool *threadpool_batch = NULL;
//		if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
//			threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
//			if (!threadpool_batch) {
//				// FIXME throw exception
//			}
//
//			// Start the non-batch threadpool in the paused state
//			tpp.paused = true;
//		}

//		struct ggml_threadpool *threadpool = ggml_threadpool_new_fn(&tpp);
		if (!threadpool) {
			threadpool = ggml_threadpool_new_fn(&tpp);
			if (!threadpool) {
				// FIXME throw exception
			}
		}

		llama_attach_threadpool(ctx, threadpool, NULL);

		return (jlong) ctx;
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return 0;
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);

	llama_detach_threadpool(ctx);
	llama_free(ctx);
}

/*
 * ACCESSORS
 */
JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doGetPoolingType(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return llama_pooling_type(ctx);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doGetContextSize(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return llama_n_ctx(ctx);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doGetBatchSize(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return llama_n_batch(ctx);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doGetPhysicalBatchSize(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return llama_n_ubatch(ctx);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doGetMaxSequenceCount(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(env, obj);
	return llama_n_seq_max(ctx);
}
