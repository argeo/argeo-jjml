#include <argeo/argeo_jni.h>
#include <bits/stdint-intn.h>
#include <ggml.h>
#include <jni.h>
#include <jni_md.h>
#include <llama.h>
#include <stddef.h>
#include <algorithm>
#include <locale>
#include <string>
#include <vector>

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
		JNIEnv *env, jobject obj, jobjectArray roles, jobjectArray contents,
		jboolean addAssistantTokens) {
	auto *model = argeo::jni::getPointer<llama_model*>(env, obj);

	const jsize messages_size = env->GetArrayLength(roles);

	std::vector<llama_chat_message> chat_messages;
	int alloc_size = 0;
	for (int i = 0; i < messages_size; i++) {
		jstring roleStr = (jstring) env->GetObjectArrayElement(roles, i);
		jboolean isCopy;
		const char *role = env->GetStringUTFChars(roleStr, &isCopy);

		jstring contentStr = (jstring) env->GetObjectArrayElement(contents, i);
		const char *content = env->GetStringUTFChars(contentStr, nullptr);

		chat_messages.push_back( { role, content });
		// using the same factor as in common.cpp
		alloc_size += (env->GetStringUTFLength(roleStr)
				+ env->GetStringUTFLength(contentStr)) * 1.25;
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
		res = llama_chat_apply_template(model, ptr_tmpl, chat_messages.data(),
				chat_messages.size(), addAssistantTokens, buf.data(),
				buf.size());
	}

	// we clean up, since we don't need the messages anymore
	for (int i = 0; i < messages_size; i++) {
		llama_chat_message message = chat_messages[i];
		jstring roleStr = (jstring) env->GetObjectArrayElement(roles, i);
		env->ReleaseStringUTFChars(roleStr, message.role);
		jstring contentStr = (jstring) env->GetObjectArrayElement(contents, i);
		env->ReleaseStringUTFChars(contentStr, message.content);
	}

	return env->NewStringUTF(buf.data());
}

/*
 * TOKENS
 */

// UTILITIES
static std::string jjml_tokens_to_cpp_string(llama_model *model,
		llama_token *tokens, int32_t n_tokens, bool remove_special,
		bool unparse_special) {
	std::string text;
	text.resize(std::max((int32_t) text.capacity(), n_tokens));
	int32_t n_chars = llama_detokenize(model, tokens, n_tokens, &text[0],
			(int32_t) text.size(), remove_special, unparse_special);
	if (n_chars < 0) {
		text.resize(-n_chars);
		n_chars = llama_detokenize(model, tokens, n_tokens, &text[0],
				(int32_t) text.size(), remove_special, unparse_special);
		GGML_ASSERT(n_chars <= (int32_t ) text.size()); // whitespace trimming is performed after per-token detokenization
	}
	text.resize(n_chars);

	return text;
}

static std::vector<llama_token> jjml_cpp_string_to_tokens(llama_model *model,
		std::string text, jboolean add_special, jboolean parse_special) {
	// upper limit for the number of tokens
	int n_tokens = text.length() + 2 * add_special;
	std::vector<llama_token> tokens(n_tokens);
	n_tokens = llama_tokenize(model, text.data(), text.length(), tokens.data(),
			tokens.size(), add_special, parse_special);
	if (n_tokens < 0) {
		tokens.resize(-n_tokens);
		int check = llama_tokenize(model, text.data(), text.length(),
				tokens.data(), tokens.size(), add_special, parse_special);
		GGML_ASSERT(check == -n_tokens);
	} else {
		tokens.resize(n_tokens);
	}

	return tokens;
}

/** The input string's byte array MUST be encoded with standard UTF-8.*/
JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doTokenizeUtf8Array(
		JNIEnv *env, jobject obj, jbyteArray str, jboolean addSpecial,
		jboolean parseSpecial) {
	auto *model = argeo::jni::getPointer<llama_model*>(env, obj);

	int text_length = env->GetArrayLength(str);
	void *arr = env->GetPrimitiveArrayCritical(str, NULL);
	char *text_chars = static_cast<char*>(arr);

	std::string text(text_chars);

	std::vector<llama_token> tokens = jjml_cpp_string_to_tokens(model, text,
			addSpecial, parseSpecial);

	// clean up
	env->ReleasePrimitiveArrayCritical(str, arr, 0);

	jintArray res = env->NewIntArray(tokens.size());
	env->SetIntArrayRegion(res, 0, tokens.size(), tokens.data());
	return res;
}

JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDeTokenizeAsUtf8Array(
		JNIEnv *env, jobject modelObj, jintArray tokenList,
		jboolean removeSpecial, jboolean unparseSpecial) {
	auto *model = argeo::jni::getPointer<llama_model*>(env, modelObj);

	int32_t n_tokens = env->GetArrayLength(tokenList);
	llama_token tokens[n_tokens];
	env->GetIntArrayRegion(tokenList, 0, n_tokens, tokens);
//	jboolean *is_copy;
//	llama_token *tokens =
//			static_cast<llama_token*>(env->GetPrimitiveArrayCritical(tokenList,
//					is_copy));

	std::string text = jjml_tokens_to_cpp_string(model, tokens, n_tokens,
			removeSpecial, unparseSpecial);

	// clean up
//	env->ReleasePrimitiveArrayCritical(tokenList, tokens, 0);

	jbyteArray res = env->NewByteArray(text.size());
	env->SetByteArrayRegion(res, 0, text.size(),
			reinterpret_cast<jbyte*>(text.data()));
	return res;
}

JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doTokenizeString(
		JNIEnv *env, jobject obj, jstring str, jboolean addSpecial,
		jboolean parseSpecial) {
	auto *model = argeo::jni::getPointer<llama_model*>(env, obj);

	const jchar *jchars = env->GetStringCritical(str, nullptr);
	std::u16string u16text = argeo::jni::jcharsToUtf16(jchars);
	std::string text = utf16_converter.to_bytes(u16text);

	std::vector<llama_token> tokens = jjml_cpp_string_to_tokens(model, text,
			addSpecial, parseSpecial);

	// clean up
	env->ReleaseStringCritical(str, jchars);

	jintArray res = env->NewIntArray(tokens.size());
	env->SetIntArrayRegion(res, 0, tokens.size(), tokens.data());
	return res;
}

JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDeTokenizeAsString(
		JNIEnv *env, jobject modelObj, jintArray tokenList,
		jboolean removeSpecial, jboolean unparseSpecial) {
	auto *model = argeo::jni::getPointer<llama_model*>(env, modelObj);

	int32_t n_tokens = env->GetArrayLength(tokenList);
	llama_token tokens[n_tokens];
	env->GetIntArrayRegion(tokenList, 0, n_tokens, tokens);
//	int32_t n_tokens = env->GetArrayLength(tokenList);
//	jboolean *is_copy;
//	jint *arr = env->GetIntArrayElements(tokenList, is_copy);
//	llama_token *tokens = static_cast<llama_token*>(arr);

	std::string text = jjml_tokens_to_cpp_string(model, tokens, n_tokens,
			removeSpecial, unparseSpecial);

	// clean up
//	env->ReleaseIntArrayElements(tokenList, arr, 0);

	std::u16string u16text = utf16_converter.from_bytes(text);
	return argeo::jni::utf16ToJstring(env, u16text);
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

/** @brief Set model parameters from native to Java.*/
//static void set_model_params(JNIEnv *env, jobject modelParams,
//		llama_model_params mparams) {
//	env->SetIntField(modelParams, LlamaCppModelParams$gpuLayerCount,
//			mparams.n_gpu_layers);
//	env->SetBooleanField(modelParams, LlamaCppModelParams$vocabOnly,
//			mparams.vocab_only);
//	env->SetBooleanField(modelParams, LlamaCppModelParams$useMlock,
//			mparams.use_mlock);
//}
/** @brief Get model parameters from Java to native.*/
static void get_model_params(JNIEnv *env, jobject params,
		llama_model_params *mparams) {
	jclass clss = env->FindClass((JNI_PKG + "LlamaCppModel$Params").c_str());
	mparams->n_gpu_layers = env->CallIntMethod(params,
			env->GetMethodID(clss, "n_gpu_layers", "()I"));
	mparams->vocab_only = env->CallIntMethod(params,
			env->GetMethodID(clss, "vocab_only", "()Z"));
	mparams->use_mlock = env->CallIntMethod(params,
			env->GetMethodID(clss, "use_mlock", "()Z"));
//	mparams->n_gpu_layers = env->GetIntField(modelParams,
//			LlamaCppModelParams$gpuLayerCount);
//	mparams->vocab_only = env->GetBooleanField(modelParams,
//			LlamaCppModelParams$vocabOnly);
//	mparams->use_mlock = env->GetBooleanField(modelParams,
//			LlamaCppModelParams$useMlock);
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llama_LlamaCppNative_newModelParams(
		JNIEnv *env, jclass) {
	llama_model_params mparams = llama_model_default_params();

	jclass clss = env->FindClass((JNI_PKG + "LlamaCppModel$Params").c_str());
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
		JNIEnv *env, jobject obj, jstring localPath, jobject modelParams,
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
//			auto *cb = (argeo::jni::java_callback*) user_data;
			// While the callback is called from the loading thread, this may change in the future,
			// so we make sure that we have a valid JNIEnv for the calling thread.
			// Note: we therefore currently don't bother detaching the thread later on.
//			JNIEnv *threadEnv;
//			cb->jvm->AttachCurrentThreadAsDaemon((void**) &threadEnv, nullptr);

//			return threadEnv->CallBooleanMethod(cb->callback, cb->method,
//					static_cast<double>(progress));
			return argeo::jni::exec_boolean_callback(
					static_cast<argeo::jni::java_callback*>(user_data),
					static_cast<jdouble>(progress));
		};
	}

	llama_model *model = llama_load_model_from_file(path_model, mparams);

	// free callback clobal reference
	if (progress_data.callback != nullptr)
		env->DeleteGlobalRef(progress_data.callback);

	env->ReleaseStringUTFChars(localPath, path_model);
	return (jlong) model;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDestroy(
		JNIEnv *env, jobject, jlong pointer) {
	auto *model = argeo::jni::getPointer<llama_model*>(pointer);
	llama_free_model(model);
}

