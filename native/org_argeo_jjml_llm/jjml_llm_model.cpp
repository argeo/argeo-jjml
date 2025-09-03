#include <stddef.h>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "org_argeo_jjml_llm_LlamaCppModel.h" // IWYU pragma: keep
#include "org_argeo_jjml_llm_LlamaCppBackend.h" // IWYU pragma: keep

#include "org_argeo_jjml_llm_.h"

// CONSTANTS
static const size_t META_BUFFER_SIZE = 1024;
static const size_t META_BIG_BUFFER_SIZE = 20480;

/*
 * PARAMETERS
 */
/** @brief Get model parameters from Java to native.*/
static void get_model_params(JNIEnv *env, jobject params,
		llama_model_params *mparams) {
	jclass clss = env->FindClass(JCLASS_MODEL_PARAMS.c_str());
	mparams->n_gpu_layers = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_gpu_layers", "()I"));
	mparams->vocab_only = env->CallIntMethod(params,
			env->GetMethodID(clss, "vocab_only", "()Z"));
	mparams->use_mmap = env->CallIntMethod(params,
			env->GetMethodID(clss, "use_mmap", "()Z"));
	mparams->use_mlock = env->CallIntMethod(params,
			env->GetMethodID(clss, "use_mlock", "()Z"));
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llm_LlamaCppBackend_newModelParams(
		JNIEnv *env, jclass) {
	llama_model_params mparams = llama_model_default_params();

	jobject res = env->NewObject(
			argeo::jni::find_jclass(env, JCLASS_MODEL_PARAMS), //
			ModelParams__init, //
			mparams.n_gpu_layers, //
			mparams.vocab_only, //
			mparams.use_mmap, //
			mparams.use_mlock //
			);
	//set_model_params(env, res, default_mparams);
	return res;
}

/*
 * LIFECYCLE
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doInit(
		JNIEnv *env, jclass, jstring localPath, jobject modelParams,
		jobject progressCallback) {
	const char *path_model = env->GetStringUTFChars(localPath, nullptr);

	llama_model_params mparams = llama_model_default_params();
	get_model_params(env, modelParams, &mparams);

	// progress callback
	argeo::jni::java_callback progress_data;
	if (progressCallback != nullptr) {
		progress_data.callback = env->NewGlobalRef(progressCallback);
		progress_data.method = DoublePredicate__test;
		env->GetJavaVM(&progress_data.jvm);
		mparams.progress_callback_user_data = &progress_data;

		mparams.progress_callback = [](float progress,
				void *user_data) -> bool {
			return argeo::jni::exec_boolean_callback(
					static_cast<argeo::jni::java_callback*>(user_data),
					static_cast<jdouble>(progress));
		};
	}

	llama_model *model = llama_model_load_from_file(path_model, mparams);

	// free callback global reference
	if (progress_data.callback != nullptr)
		env->DeleteGlobalRef(progress_data.callback);

	env->ReleaseStringUTFChars(localPath, path_model);
	return (jlong) model;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	llama_model_free(model);
}

/*
 * ACCESSORS
 */
JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetVocabularySize(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	const llama_vocab *vocab = llama_model_get_vocab(model);
	return llama_vocab_n_tokens(vocab);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetContextTrainingSize(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_model_n_ctx_train(model);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetEmbeddingSize(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_model_n_embd(model);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetLayerCount(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_model_n_layer(model);
}

/** Gather metadata keys or values. */
static jobjectArray jjml_lama_get_meta(JNIEnv *env, llama_model *model,
		std::function<int32_t(int32_t, char*, size_t)> supplier) {
	try {
		int32_t meta_count = llama_model_meta_count(model);

		jobjectArray res = env->NewObjectArray(meta_count, env->FindClass("[B"),
				nullptr);
		for (int32_t i = 0; i < meta_count; i++) {
			try {

				char buf[META_BUFFER_SIZE];
				int32_t length = supplier(i, buf, META_BUFFER_SIZE);
				if (length == -1)
					throw std::runtime_error(
							"Cannot read model metadata " + std::to_string(i));
				std::string u8_res;
				if (length > META_BUFFER_SIZE) { // chat templates can be quite big
					char big_buf[META_BIG_BUFFER_SIZE];
					length = supplier(i, big_buf, length);
					u8_res = std::string(big_buf, length);
				} else {
					u8_res = std::string(buf, length);
				}
				jbyteArray str = env->NewByteArray(u8_res.length());
				env->SetObjectArrayElement(res, i, str);
				env->SetByteArrayRegion(str, 0, u8_res.length(),
						(jbyte*) u8_res.c_str());
			} catch (std::exception &ex) {
				// ignore
				std::cerr << "Cannot read metadata " << i << ": " << ex.what()
						<< ". Ignoring it." << std::endl;
			}
		}
		return res;
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

JNIEXPORT jobjectArray JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetMetadataKeys(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return jjml_lama_get_meta(env, model,
			[model](int32_t i, char *buf, size_t buf_size) {
				return llama_model_meta_key_by_index(model, i, buf, buf_size);
			});
}

JNIEXPORT jobjectArray JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetMetadataValues(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return jjml_lama_get_meta(env, model,
			[model](int32_t i, char *buf, size_t buf_size) {
				return llama_model_meta_val_str_by_index(model, i, buf,
						buf_size);
			});
}

JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetDescription(
		JNIEnv *env, jobject obj) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
		char buf[META_BUFFER_SIZE];
		int32_t length = llama_model_desc(model, buf, META_BUFFER_SIZE);
		if (length == -1)
			throw std::runtime_error("Cannot read model description ");
		std::string u8_res;
		if (length > META_BUFFER_SIZE) { // big description
			char big_buf[META_BIG_BUFFER_SIZE];
			length = llama_model_desc(model, big_buf, length);
			u8_res = std::string(big_buf, length);
		} else {
			u8_res = std::string(buf, length);
		}
		jbyteArray res = env->NewByteArray(u8_res.length());
		env->SetByteArrayRegion(res, 0, u8_res.length(),
				(jbyte*) u8_res.c_str());
		return res;
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetModelSize(
		JNIEnv *env, jobject obj) {
	static_assert(sizeof(jlong) >= sizeof(uint64_t));
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_model_size(model);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llm_LlamaCppModel_doGetEndOfGenerationToken(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	const llama_vocab *vocab = llama_model_get_vocab(model);
	llama_token eot = llama_vocab_eot(vocab);
	return eot == -1 ? llama_vocab_eos(vocab) : eot;
}
