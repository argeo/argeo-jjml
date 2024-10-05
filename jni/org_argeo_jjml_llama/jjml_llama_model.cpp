#include <string>
#include <vector>

#include <llama.h>

#include "argeo_jni_utils.h"
#include "jjml_llama.h"

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppModel.h"
#include "org_argeo_jjml_llama_LlamaCppNative.h"

/*
 * CHAT
 */
JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doFormatChatMessages(
		JNIEnv *env, jobject obj, jobjectArray roles, jobjectArray contents) {
	llama_model *model = (llama_model*) getPointer(env, obj);

	const jsize messages_size = env->GetArrayLength(roles);

	std::vector<llama_chat_message> chat_messages;
	int alloc_size = 0;
	for (int i = 0; i < messages_size; i++) {
		jstring roleStr = (jstring) env->GetObjectArrayElement(roles, i);
		jboolean isCopy;
		const char *role = env->GetStringUTFChars(roleStr, &isCopy);
//		std::cerr << role << isCopy << "\n";

		jstring contentStr = (jstring) env->GetObjectArrayElement(contents, i);
		const char *content = env->GetStringUTFChars(contentStr, nullptr);
//		std::cerr << content << "\n";

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
			ThrowIllegalStateException(env, "Custom template is not supported");
		} else {
			ThrowIllegalStateException(env,
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
	jboolean *is_copy;
	void *region = env->GetPrimitiveArrayCritical(tokens, is_copy);

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
 * ACCESSORS
 */
JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doGetEmbeddingSize(
		JNIEnv *env, jobject obj) {
	llama_model *model = (llama_model*) getPointer(env, obj);
	return llama_n_embd(model);
}

/*
 * PARAMETERS
 */

/** @brief Set model parameters from native to Java.*/
static void set_model_params(JNIEnv *env, jobject modelParams,
		llama_model_params mparams) {
	env->SetIntField(modelParams, LlamaCppModelParams$gpuLayerCount(env),
			mparams.n_gpu_layers);
	env->SetBooleanField(modelParams, LlamaCppModelParams$vocabOnly(env),
			mparams.vocab_only);
	env->SetBooleanField(modelParams, LlamaCppModelParams$useMlock(env),
			mparams.use_mlock);
}

/** @brief Get model parameters from Java to native.*/
static void get_model_params(JNIEnv *env, jobject modelParams,
		llama_model_params *mparams) {
	mparams->n_gpu_layers = env->GetIntField(modelParams,
			LlamaCppModelParams$gpuLayerCount(env));
	mparams->vocab_only = env->GetBooleanField(modelParams,
			LlamaCppModelParams$vocabOnly(env));
	mparams->use_mlock = env->GetBooleanField(modelParams,
			LlamaCppModelParams$useMlock(env));
}

JNIEXPORT jobject JNICALL Java_org_argeo_jjml_llama_LlamaCppNative_newModelParams(
		JNIEnv *env, jclass) {
	jobject res = env->NewObject(LlamaCppModelParams(env),
			LlamaCppModelParams$LlamaCppModelParams(env));
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
					DoublePredicate$test(threadEnv),
					static_cast<double>(progress));
		};
	}

	llama_model *model = llama_load_model_from_file(path_model, mparams);

	env->ReleaseStringUTFChars(localPath, path_model);
	return (jlong) model;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDestroy(
		JNIEnv *env, jobject obj) {
	llama_model *model = (llama_model*) getPointer(env, obj);
	llama_free_model(model);
}

