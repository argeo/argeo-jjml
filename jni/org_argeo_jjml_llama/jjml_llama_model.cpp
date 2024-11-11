#include <argeo/argeo_jni.h>
#include <ggml.h>
#include <jni.h>
#include <jni_md.h>
#include <llama.h>
#include <stddef.h>
#include <algorithm>
#include <locale>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppModel.h" // IWYU pragma: keep
#include "org_argeo_jjml_llama_LlamaCppNative.h" // IWYU pragma: keep

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
	auto *model = argeo::jni::getPointer<llama_model*>(pointer);

	const jsize messages_size = env->GetArrayLength(roles);
	std::vector<llama_chat_message> chat_messages;

	try {
		int alloc_size = 0;
		for (int i = 0; i < messages_size; i++) {
			jstring roleStr = (jstring) env->GetObjectArrayElement(roles, i);
			std::u16string u16role = argeo::jni::jstringToUtf16(env, roleStr);
			std::string u8role = utf16_converter.to_bytes(u16role);
			char *role = new char[u8role.length()];
			strcpy(role, u8role.c_str());

			jstring contentStr = (jstring) env->GetObjectArrayElement(contents,
					i);
			std::u16string u16content = argeo::jni::jstringToUtf16(env,
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
		int32_t res = llama_chat_apply_template(model, ptr_tmpl,
				chat_messages.data(), chat_messages.size(), addAssistantTokens,
				buf.data(), buf.size());

		// error: chat template is not supported
		if (res < 0) {
			if (ptr_tmpl != nullptr) {
				env->ThrowNew(IllegalStateException,
						"Custom template is not supported");
			} else {
				env->ThrowNew(IllegalStateException,
						"Built-in template is not supported");
			}
			return nullptr;
		}

		// if it turns out that our buffer is too small, we resize it
		if ((size_t) res > buf.size()) {
			buf.resize(res);
			res = llama_chat_apply_template(model, ptr_tmpl,
					chat_messages.data(), chat_messages.size(),
					addAssistantTokens, buf.data(), buf.size());
		}

		// we clean up, since we don't need the messages anymore
		for (int i = 0; i < messages_size; i++) {
			llama_chat_message message = chat_messages[i];
			delete message.role;
			delete message.content;
//		jstring roleStr = (jstring) env->GetObjectArrayElement(roles, i);
//		env->ReleaseStringUTFChars(roleStr, message.role);
//		jstring contentStr = (jstring) env->GetObjectArrayElement(contents, i);
//		env->ReleaseStringUTFChars(contentStr, message.content);
		}

		std::string u8res(buf.data());
		u8res.resize(res);
		std::u16string u16res = utf16_converter.from_bytes(u8res);
		return argeo::jni::utf16ToJstring(env, u16res);
	} catch (std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
//	return env->NewStringUTF(buf.data());
}

/*
 * ACCESSORS
 */
JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doGetEmbeddingSize(
		JNIEnv *env, jobject obj) {
	auto *model = argeo::jni::getPointer<llama_model*>(env, obj);
	return llama_n_embd(model);
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
	mparams->use_mlock = env->CallIntMethod(params,
			env->GetMethodID(clss, "use_mlock", "()Z"));
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llama_LlamaCppNative_newModelParams(
		JNIEnv *env, jclass) {
	llama_model_params mparams = llama_model_default_params();

	jclass clss = env->FindClass(JCLASS_MODEL_PARAMS.c_str());
	jmethodID constructor = env->GetMethodID(clss, "<init>", "(IZZ)V");
	jobject res = env->NewObject(clss, constructor, //
			mparams.n_gpu_layers, //
			mparams.vocab_only, //
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
	auto *model = argeo::jni::getPointer<llama_model*>(env, obj);
	llama_free_model(model);
}

