#include <argeo/argeo_jni.h>
#include <ggml.h>
#include <jni.h>
#include <jni_md.h>
#include <llama.h>
#include <stddef.h>
#include <algorithm>
#include <locale>
#include <string>
#include <cstring>
#include <vector>
#include <cassert>

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppVocabulary.h" // IWYU pragma: keep

/*
 * FIELDS
 */
/** UTF-16 converter. */
static argeo::jni::utf16_convert utf16_converter;

/*
 * VOCABULARY
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

/*
 * UTF-8 from Java
 */

/** The input string's byte array MUST be encoded with standard UTF-8.*/
JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doTokenizeUtf8BytesAsArray(
		JNIEnv *env, jobject, jlong pointer, jbyteArray str, jint offset,
		jint length, jboolean addSpecial, jboolean parseSpecial) {
	auto *model = argeo::jni::getPointer<llama_model*>(pointer);

//	int text_length = env->GetArrayLength(str);
//	void *arr = env->GetPrimitiveArrayCritical(str, NULL);
//	char *text_chars = static_cast<char*>(arr) + offset;
//	std::string text(text_chars, length);

	char arr[length];
	env->GetByteArrayRegion(str, offset, length, reinterpret_cast<jbyte*>(arr));
	std::string text(arr, length);

	std::vector<llama_token> tokens = jjml_cpp_string_to_tokens(model, text,
			addSpecial, parseSpecial);

	// clean up
//	env->ReleasePrimitiveArrayCritical(str, arr, 0);

	jintArray res = env->NewIntArray(tokens.size());
	env->SetIntArrayRegion(res, 0, tokens.size(), reinterpret_cast<jint*>(tokens.data()));
	return res;
}

JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doTokenizeUtf8AsArray(
		JNIEnv*, jobject, jlong, jobject, jboolean, jboolean) {
	// TODO
	return nullptr;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doTokenizeUtf8Bytes(
		JNIEnv *env, jobject, jlong pointer, jbyteArray str, jint offset,
		jint length, jobject buf, jboolean addSpecial, jboolean parseSpecial) {
	try {
		auto *model = argeo::jni::getPointer<llama_model*>(pointer);

		int tokens_size = env->CallIntMethod(buf, IntBuffer$limit);
		llama_token *output =
				static_cast<llama_token*>(env->GetDirectBufferAddress(buf));

		void *arr = env->GetPrimitiveArrayCritical(str, NULL);
		char *text_chars = static_cast<char*>(arr) + offset;
//		std::string text(text_chars, length);

		int n_tokens = llama_tokenize(model, text_chars, length, output,
				tokens_size, addSpecial, parseSpecial);

		// clean up
		env->ReleasePrimitiveArrayCritical(str, arr, 0);

		if (n_tokens < 0) {
			throw std::range_error(
					"Input buffer limit " + std::to_string(tokens_size)
							+ " is too small for " + std::to_string(-n_tokens)
							+ " tokens - " + __func__);
		}

		// set buffer in a proper state
		env->CallVoidMethod(buf, IntBuffer$limitI, n_tokens);
		env->CallVoidMethod(buf, IntBuffer$positionI, n_tokens);
	} catch (const std::exception &ex) {
		env->ThrowNew(IllegalStateException, ex.what());
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doTokenizeUtf8(
		JNIEnv*, jobject, jlong, jobject, jobject, jboolean, jboolean) {

}

JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeArrayAsUtf8Bytes(
		JNIEnv *env, jobject, jlong pointer, jintArray tokenList, jint offset,
		jint length, jboolean removeSpecial, jboolean unparseSpecial) {
	auto *model = argeo::jni::getPointer<llama_model*>(pointer);

	int32_t n_tokens = env->GetArrayLength(tokenList);
	llama_token tokens[n_tokens];
	env->GetIntArrayRegion(tokenList, 0, n_tokens, reinterpret_cast<jint*>(tokens));
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

JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeAsUtf8Bytes(
		JNIEnv*, jobject, jlong, jobject, jboolean, jboolean) {
	// TODO
	return nullptr;
}

static void jjml_llama_detokenize(JNIEnv *env, llama_model *model,
		llama_token *tokens, int tokens_size, jobject outBuf,
		bool remove_special, bool unparse_special) {
	int out_size = env->CallIntMethod(outBuf, IntBuffer$limit);
	char *out = static_cast<char*>(env->GetDirectBufferAddress(outBuf));
	int32_t n_chars = llama_detokenize(model, tokens, tokens_size, out,
			out_size, remove_special, unparse_special);

	if (n_chars < 0)
		throw std::range_error(
				"Output buffer capacity " + std::to_string(out_size)
						+ " is too small for " + std::to_string(-n_chars)
						+ " characters - " + __func__);
//		std::memcpy(out, text.data(), text.length());

	// set buffer into a proper state
	env->CallVoidMethod(outBuf, IntBuffer$positionI, n_chars);
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeArrayAsUtf8(
		JNIEnv* env, jobject, jlong pointer, jintArray tokensArray, jint offset, jint tokens_size, jobject outBuf, jboolean removeSpecial,
		jboolean unparseSpecial) {
	try {
		auto *model = argeo::jni::getPointer<llama_model*>(pointer);

		void* arr = env->GetPrimitiveArrayCritical(tokensArray,
				nullptr);
			llama_token *tokens =
					static_cast<llama_token*>(arr);

		jjml_llama_detokenize(env, model, tokens, tokens_size, outBuf,
				removeSpecial, unparseSpecial);

		env->ReleasePrimitiveArrayCritical(tokensArray, arr, 0);

	} catch (const std::exception &ex) {
		env->ThrowNew(IllegalStateException, ex.what());
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeAsUtf8(
		JNIEnv *env, jobject, jlong pointer, jobject buf, jobject outBuf,
		jboolean removeSpecial, jboolean unparseSpecial) {
	try {
		auto *model = argeo::jni::getPointer<llama_model*>(pointer);

		int tokens_size = env->CallIntMethod(buf, IntBuffer$limit);
		llama_token *tokens =
				static_cast<llama_token*>(env->GetDirectBufferAddress(buf));

		jjml_llama_detokenize(env, model, tokens, tokens_size, outBuf,
				removeSpecial, unparseSpecial);

//		int out_size = env->CallIntMethod(outBuf, IntBuffer$limit);
//		char *out = static_cast<char*>(env->GetDirectBufferAddress(outBuf));
//
//		int32_t n_chars = llama_detokenize(model, tokens, tokens_size, out,
//				out_size, removeSpecial, unparseSpecial);
//
//		if (n_chars < 0)
//			throw std::range_error(
//					"Output buffer capacity " + std::to_string(out_size)
//							+ " is too small for " + std::to_string(-n_chars)
//							+ " characters - " + __func__);
//
//		// set buffer into a proper state
//		env->CallVoidMethod(outBuf, IntBuffer$positionI, n_chars);
	} catch (const std::exception &ex) {
		env->ThrowNew(IllegalStateException, ex.what());
	}

}

JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doTokenizeStringAsArray(
		JNIEnv *env, jobject, jlong pointer, jstring str, jboolean addSpecial,
		jboolean parseSpecial) {
	auto *model = argeo::jni::getPointer<llama_model*>(pointer);

	const jchar *jchars = env->GetStringCritical(str, nullptr);
	std::u16string u16text = argeo::jni::jcharsToUtf16(jchars);
	std::string text = utf16_converter.to_bytes(u16text);

	std::vector<llama_token> tokens = jjml_cpp_string_to_tokens(model, text,
			addSpecial, parseSpecial);

	// clean up
	env->ReleaseStringCritical(str, jchars);

	jintArray res = env->NewIntArray(tokens.size());
	env->SetIntArrayRegion(res, 0, tokens.size(), reinterpret_cast<jint*>(tokens.data()));
	return res;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doTokenizeString(
		JNIEnv *env, jobject, jlong pointer, jstring str, jobject buf,
		jboolean addSpecial, jboolean parseSpecial) {
	try {
		auto *model = argeo::jni::getPointer<llama_model*>(pointer);
		assert(
				(env->CallIntMethod(buf, IntBuffer$position) == 0
						&& "Buffer position must be 0"));
		int tokens_size = env->CallIntMethod(buf, IntBuffer$limit);
		llama_token *tokens =
				static_cast<llama_token*>(env->GetDirectBufferAddress(buf));

		// BEGIN STRING CRITICAL
		const jchar *jchars = env->GetStringCritical(str, nullptr);
		std::u16string u16text = argeo::jni::jcharsToUtf16(jchars);
		std::string text = utf16_converter.to_bytes(u16text);

		int n_tokens = llama_tokenize(model, text.data(), text.length(), tokens,
				tokens_size, addSpecial, parseSpecial);
		if (n_tokens < 0) {
			throw std::range_error(
					"Input buffer limit " + std::to_string(tokens_size)
							+ " is too small for " + std::to_string(-n_tokens)
							+ " tokens - " + __func__);
		}

		// clean up
		env->ReleaseStringCritical(str, jchars);
		// END STRING CRITICAL

		// set buffer in a proper state
		env->CallVoidMethod(buf, IntBuffer$limitI, n_tokens);
		env->CallVoidMethod(buf, IntBuffer$positionI, n_tokens);
	} catch (const std::exception &ex) {
		env->ThrowNew(IllegalStateException, ex.what());
	}
}

JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeArrayAsString(
		JNIEnv *env, jobject, jlong pointer, jintArray tokenList, jint offset,
		jint length, jboolean removeSpecial, jboolean unparseSpecial) {
	auto *model = argeo::jni::getPointer<llama_model*>(pointer);

//	int32_t n_tokens = env->GetArrayLength(tokenList);
	int32_t n_tokens = length;
	llama_token tokens[n_tokens];
	env->GetIntArrayRegion(tokenList, offset, n_tokens, reinterpret_cast<jint*>(tokens));
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

JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeAsString(
		JNIEnv*, jobject, jlong, jobject, jboolean, jboolean) {
	// TODO
	return nullptr;
}

