#include <stddef.h>
#include <cstring>
#include <functional>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

#include <llama.h>

#include <argeo/argeo_jni.h>

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppModel.h" // IWYU pragma: keep
#include "org_argeo_jjml_llama_LlamaCppBackend.h" // IWYU pragma: keep

/*
 * FIELDS
 */
/** UTF-16 converter. */
static argeo::jni::utf16_convert utf16_converter;

/*
 * CHAT
 */
JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doFormatChatMessages(
		JNIEnv *env, jobject, jlong pointer, jobjectArray roles,
		jobjectArray contents, jboolean addAssistantTokens) {
	auto *model = argeo::jni::as_pointer<llama_model*>(pointer);

	const jsize messages_size = env->GetArrayLength(roles);
	std::vector<llama_chat_message> chat_messages;

	try {
		int alloc_size = 0;
		for (int i = 0; i < messages_size; i++) {
			jstring roleStr = (jstring) env->GetObjectArrayElement(roles, i);
			std::u16string u16role = argeo::jni::jstring_to_utf16(env, roleStr);
			std::string u8role = utf16_converter.to_bytes(u16role);
			char *role = new char[u8role.length()];
			strcpy(role, u8role.c_str());

			jstring contentStr = (jstring) env->GetObjectArrayElement(contents,
					i);
			std::u16string u16content = argeo::jni::jstring_to_utf16(env,
					contentStr);
			std::string u8content = utf16_converter.to_bytes(u16content);
			char *content = new char[u8content.length()];
			strcpy(content, u8content.c_str());

			llama_chat_message message { role, content };
			chat_messages.push_back(message);
			// using the same factor as in common.cpp
			alloc_size += (u8role.length() + u8content.length()) * 1.25;
		}

		const char *ptr_tmpl = nullptr; // TODO custom template
		std::vector<char> buf(alloc_size);
		int32_t resLength = llama_chat_apply_template(model, ptr_tmpl,
				chat_messages.data(), chat_messages.size(), addAssistantTokens,
				buf.data(), buf.size());

		// error: chat template is not supported
		if (resLength < 0) {
			if (ptr_tmpl != nullptr)
				throw std::runtime_error("Custom template is not supported");
			else
				throw std::runtime_error("Built-in template is not supported");
		}

		// if it turns out that our buffer is too small, we resize it
		if ((size_t) resLength > buf.size()) {
			buf.resize(resLength);
			resLength = llama_chat_apply_template(model, ptr_tmpl,
					chat_messages.data(), chat_messages.size(),
					addAssistantTokens, buf.data(), buf.size());
		}

		// we clean up, since we don't need the messages anymore
		for (int i = 0; i < messages_size; i++) {
			llama_chat_message message = chat_messages[i];
			delete message.role;
			delete message.content;
		}

		std::u16string u16res = utf16_converter.from_bytes(
				std::string(buf.data(), resLength));
		return argeo::jni::utf16_to_jstring(env, u16res);
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

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

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_newModelParams(
		JNIEnv *env, jclass) {
	llama_model_params mparams = llama_model_default_params();

	jobject res = env->NewObject(
			argeo::jni::find_jclass(env, JCLASS_MODEL_PARAMS), //
			ModelParams$init, //
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
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doInit(
		JNIEnv *env, jclass, jstring localPath, jobject modelParams,
		jobject progressCallback) {
	const char *path_model = env->GetStringUTFChars(localPath, nullptr);

	llama_model_params mparams = llama_model_default_params();
	get_model_params(env, modelParams, &mparams);

	// progress callback
	argeo::jni::java_callback progress_data;
	if (progressCallback != nullptr) {
		progress_data.callback = env->NewGlobalRef(progressCallback);
		progress_data.method = DoublePredicate$test;
		env->GetJavaVM(&progress_data.jvm);
		mparams.progress_callback_user_data = &progress_data;

		mparams.progress_callback = [](float progress,
				void *user_data) -> bool {
			return argeo::jni::exec_boolean_callback(
					static_cast<argeo::jni::java_callback*>(user_data),
					static_cast<jdouble>(progress));
		};
	}

	llama_model *model = llama_load_model_from_file(path_model, mparams);

	// free callback global reference
	if (progress_data.callback != nullptr)
		env->DeleteGlobalRef(progress_data.callback);

	env->ReleaseStringUTFChars(localPath, path_model);
	return (jlong) model;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	llama_free_model(model);
}

/*
 * ACCESSORS
 */
JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doGetVocabularySize(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_n_vocab(model);

}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doGetContextTrainingSize(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_n_ctx_train(model);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doGetEmbeddingSize(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_n_embd(model);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doGetLayerCount(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return llama_n_layer(model);
}

static jobjectArray jjml_lama_get_meta(JNIEnv *env, llama_model *model,
		std::function<int32_t(int32_t, char*, size_t)> supplier) {
	try {
		int32_t meta_count = llama_model_meta_count(model);

		jobjectArray res = env->NewObjectArray(meta_count,
				env->FindClass("java/lang/String"), nullptr);
		for (int32_t i = 0; i < meta_count; i++) {
			try {
				// chat templates can be big
				const size_t buf_size = 1024;
				char buf[buf_size];
				int32_t length = supplier(i, buf, buf_size);
				if (length == -1)
					throw std::runtime_error(
							"Cannot read model metadata " + std::to_string(i));
				std::u16string u16res;
				if (length > buf_size) { // chat templates can be quite big
					char big_buf[length];
					length = supplier(i, big_buf, length);
					u16res = utf16_converter.from_bytes(
							std::string(big_buf, length));
				} else {
					u16res = utf16_converter.from_bytes(
							std::string(buf, length));
				}
				jstring str = argeo::jni::utf16_to_jstring(env, u16res);
				env->SetObjectArrayElement(res, i, str);
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

JNIEXPORT jobjectArray JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doGetMetadataKeys(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return jjml_lama_get_meta(env, model,
			[model](int32_t i, char *buf, size_t buf_size) {
				return llama_model_meta_key_by_index(model, i, buf, buf_size);
			});
}

JNIEXPORT jobjectArray JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doGetMetadataValues(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
	return jjml_lama_get_meta(env, model,
			[model](int32_t i, char *buf, size_t buf_size) {
				return llama_model_meta_val_str_by_index(model, i, buf,
						buf_size);
			});
}

JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doGetDescription(
		JNIEnv *env, jobject obj) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(env, obj);
		const size_t buf_size = 1024;
		char buf[buf_size];
		int32_t length = llama_model_desc(model, buf, buf_size);
		if (length == -1)
			throw std::runtime_error("Cannot read model description ");
		std::u16string u16res;
		if (length > buf_size) { // big description
			char big_buf[length];
			length = llama_model_desc(model, big_buf, length);
			u16res = utf16_converter.from_bytes(std::string(big_buf, length));
		} else {
			u16res = utf16_converter.from_bytes(std::string(buf, length));
		}
		return argeo::jni::utf16_to_jstring(env, u16res);
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}
