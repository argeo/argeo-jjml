#include <argeo/argeo_jni.h>
#include <bits/stdint-intn.h>
#include <bits/types/mbstate_t.h>
#include <ggml.h>
#include <jni.h>
#include <jni_md.h>
#include <llama.h>
#include <stddef.h>
#include <algorithm>
#include <fstream>
#include <locale>
#include <string>
#include <vector>

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppModel.h" // IWYU pragma: keep
#include "org_argeo_jjml_llama_LlamaCppNative.h" // IWYU pragma: keep

std::wstring_convert<
		argeo::jni::deletable_facet<std::codecvt<char16_t, char, std::mbstate_t>>,
		char16_t> convertUtf16;

/*
 * CHAT
 */
JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doFormatChatMessages(
		JNIEnv *env, jobject obj, jobjectArray roles, jobjectArray contents) {
	auto *model = getPointer<llama_model*>(env, obj);

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
			chat_messages.data(), chat_messages.size(), true, buf.data(),
			buf.size());

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
				chat_messages.size(), true, buf.data(), buf.size());
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
	auto *model = getPointer<llama_model*>(env, obj);

	int text_length = env->GetArrayLength(str);
	char *text_chars = static_cast<char*>(env->GetPrimitiveArrayCritical(str,
	NULL));

	std::string text(text_chars);

	std::vector<llama_token> tokens = jjml_cpp_string_to_tokens(model, text,
			addSpecial, parseSpecial);

	// clean up
	env->ReleasePrimitiveArrayCritical(str, text_chars, 0);

	jintArray res = env->NewIntArray(tokens.size());
	env->SetIntArrayRegion(res, 0, tokens.size(), tokens.data());
	return res;
}

JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDeTokenizeAsUtf8Array(
		JNIEnv *env, jobject modelObj, jintArray tokenList,
		jboolean removeSpecial, jboolean unparseSpecial) {
	auto *model = getPointer<llama_model*>(env, modelObj);

	int32_t n_tokens = env->GetArrayLength(tokenList);
	jboolean *is_copy;
	llama_token *tokens =
			static_cast<llama_token*>(env->GetPrimitiveArrayCritical(tokenList,
					is_copy));

	std::string text = jjml_tokens_to_cpp_string(model, tokens, n_tokens,
			removeSpecial, unparseSpecial);

	// clean up
	env->ReleasePrimitiveArrayCritical(tokenList, tokens, 0);

	jbyteArray res = env->NewByteArray(text.size());
	env->SetByteArrayRegion(res, 0, text.size(),
			reinterpret_cast<jbyte*>(text.data()));
	return res;
}

JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doTokenizeString(
		JNIEnv *env, jobject obj, jstring str, jboolean addSpecial,
		jboolean parseSpecial) {
	auto *model = getPointer<llama_model*>(env, obj);

	const char16_t *u16chars =
			reinterpret_cast<const char16_t*>(env->GetStringCritical(str,
					nullptr));

	std::u16string u16text(u16chars);
	std::string text = convertUtf16.to_bytes(u16text);

	std::vector<llama_token> tokens = jjml_cpp_string_to_tokens(model, text,
			addSpecial, parseSpecial);

	// clean up
	env->ReleaseStringCritical(str, reinterpret_cast<const jchar*>(u16chars));

	jintArray res = env->NewIntArray(tokens.size());
	env->SetIntArrayRegion(res, 0, tokens.size(), tokens.data());
	return res;
}

JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDeTokenizeAsString(
		JNIEnv *env, jobject modelObj, jintArray tokenList,
		jboolean removeSpecial, jboolean unparseSpecial) {

//	std::string text = reinterpret_cast<const char*>(+u8"zß水🍌");
	auto *model = getPointer<llama_model*>(env, modelObj);

	int32_t n_tokens = env->GetArrayLength(tokenList);
	jboolean *is_copy;
	llama_token *tokens =
			static_cast<llama_token*>(env->GetPrimitiveArrayCritical(tokenList,
					is_copy));

	std::string text = jjml_tokens_to_cpp_string(model, tokens, n_tokens,
			removeSpecial, unparseSpecial);

	// clean up
	env->ReleasePrimitiveArrayCritical(tokenList, tokens, 0);

	std::u16string u16text = convertUtf16.from_bytes(text);
	const jchar *resChars = reinterpret_cast<const jchar*>(u16text.data());
	jstring res = env->NewString(resChars, u16text.size());
	return res;
}

/*
 * ACCESSORS
 */
JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doGetEmbeddingSize(
		JNIEnv *env, jobject obj) {
	auto *model = getPointer<llama_model*>(env, obj);
	return llama_n_embd(model);
}

/*
 * PARAMETERS
 */

/** @brief Set model parameters from native to Java.*/
static void set_model_params(JNIEnv *env, jobject modelParams,
		llama_model_params mparams) {
	env->SetIntField(modelParams, LlamaCppModelParams$gpuLayerCount,
			mparams.n_gpu_layers);
	env->SetBooleanField(modelParams, LlamaCppModelParams$vocabOnly,
			mparams.vocab_only);
	env->SetBooleanField(modelParams, LlamaCppModelParams$useMlock,
			mparams.use_mlock);
}

/** @brief Get model parameters from Java to native.*/
static void get_model_params(JNIEnv *env, jobject modelParams,
		llama_model_params *mparams) {
	mparams->n_gpu_layers = env->GetIntField(modelParams,
			LlamaCppModelParams$gpuLayerCount);
	mparams->vocab_only = env->GetBooleanField(modelParams,
			LlamaCppModelParams$vocabOnly);
	mparams->use_mlock = env->GetBooleanField(modelParams,
			LlamaCppModelParams$useMlock);
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llama_LlamaCppNative_newModelParams(
		JNIEnv *env, jclass) {
	jobject res = env->NewObject(LlamaCppModelParams,
			LlamaCppModelParams$LlamaCppModelParams);
	llama_model_params default_mparams = llama_model_default_params();
	set_model_params(env, res, default_mparams);
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
	if (progressCallback != nullptr) {
		// struct used to pass references to the lambda
		struct java_callback {
			const jobject callback;
			JavaVM *jvm;
		} progress_data { progressCallback };
		// we may have multiple JVMs, we therefore don't centralize a static reference
		env->GetJavaVM(&progress_data.jvm);
		mparams.progress_callback_user_data = &progress_data;

		mparams.progress_callback = [](float progress,
				void *user_data) -> bool {
			java_callback *cb = (java_callback*) user_data;
			// While the callback is called from the loading thread, this may change in the future,
			// so we make sure that we have a valid JNIEnv for the calling thread.
			// Note: we therefore currently don't bother detaching the thread later on.
			JNIEnv *threadEnv;
			cb->jvm->AttachCurrentThreadAsDaemon((void**) &threadEnv, nullptr);

			return threadEnv->CallBooleanMethod(cb->callback,
					DoublePredicate$test, static_cast<double>(progress));
		};
	}

	llama_model *model = llama_load_model_from_file(path_model, mparams);

	env->ReleaseStringUTFChars(localPath, path_model);
	return (jlong) model;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *model = getPointer<llama_model*>(env, obj);
	llama_free_model(model);
}

