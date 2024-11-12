#include <algorithm>
#include <cassert>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

#include <ggml.h>
#include <llama.h>

#include <argeo/argeo_jni.h>

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
		char *u8_chars, int u8_size, jboolean add_special,
		jboolean parse_special) {
	// upper limit for the number of tokens
	int n_tokens = u8_size + 2 * add_special;
	std::vector<llama_token> tokens(n_tokens);
	n_tokens = llama_tokenize(model, u8_chars, u8_size, tokens.data(),
			tokens.size(), add_special, parse_special);
	if (n_tokens < 0) {
		tokens.resize(-n_tokens);
		int check = llama_tokenize(model, u8_chars, u8_size, tokens.data(),
				tokens.size(), add_special, parse_special);
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
		JNIEnv *env, jclass, jlong pointer, jbyteArray str, jint offset,
		jint length, jboolean addSpecial, jboolean parseSpecial) {
	auto *model = argeo::jni::as_pointer<llama_model*>(pointer);

	void *u8_arr = env->GetPrimitiveArrayCritical(str, NULL);
	char *u8_chars = static_cast<char*>(u8_arr) + offset;

	std::vector<llama_token> tokens = jjml_cpp_string_to_tokens(model, u8_chars,
			length, addSpecial, parseSpecial);

	// clean up
	env->ReleasePrimitiveArrayCritical(str, u8_arr, 0);

	jintArray res = env->NewIntArray(tokens.size());
	env->SetIntArrayRegion(res, 0, tokens.size(),
			reinterpret_cast<jint*>(tokens.data()));
	return res;
}

JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doTokenizeUtf8AsArray(
		JNIEnv *env, jclass, jlong pointer, jobject u8Buf, jint offset,
		jint length, jboolean addSpecial, jboolean parseSpecial) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(pointer);

		// input
		void *u8_arr = env->GetDirectBufferAddress(u8Buf);
		if (u8_arr == NULL)
			throw std::invalid_argument("Input is not a direct buffer");
		assert(env->GetDirectBufferCapacity(u8Buf) >= offset + length);
		char *u8_chars = static_cast<char*>(u8_arr) + offset;
//		std::string text(u8_chars, u8_size);

		std::vector<llama_token> tokens = jjml_cpp_string_to_tokens(model,
				u8_chars, length, addSpecial, parseSpecial);

		jintArray res = env->NewIntArray(tokens.size());
		env->SetIntArrayRegion(res, 0, tokens.size(),
				reinterpret_cast<jint*>(tokens.data()));

		return res;
	} catch (const std::exception &ex) {
		return argeo::jni::throw_to_java(env, ex);
	}
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doTokenizeUtf8(
		JNIEnv *env, jclass, jlong pointer, jobject u8Buf, jint offset,
		jint length, jobject tokensBuf, jint pos, jint size,
		jboolean addSpecial, jboolean parseSpecial) {
	jint n_tokens = 0;
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(pointer);

		// input
		void *u8_arr = env->GetDirectBufferAddress(u8Buf);
		if (u8_arr == NULL)
			throw std::invalid_argument("Input is not a direct buffer");
		assert(env->GetDirectBufferCapacity(u8Buf) >= offset + length);
		char *u8_chars = static_cast<char*>(u8_arr) + offset;

		// output
		void *tokens_arr = env->GetDirectBufferAddress(tokensBuf);
		if (tokens_arr == NULL)
			throw std::invalid_argument("Output is not a direct buffer");
		llama_token *tokens = static_cast<llama_token*>(tokens_arr) + pos;

		n_tokens = llama_tokenize(model, u8_chars, length, tokens, size,
				addSpecial, parseSpecial);

//		if (n_tokens < 0)
//			throw std::range_error(
//					"Input buffer limit " + std::to_string(size)
//							+ " is too small for " + std::to_string(-n_tokens)
//							+ " tokens - " + __func__);
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
	}
	return n_tokens;
}

JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeArrayAsUtf8Bytes(
		JNIEnv *env, jclass, jlong pointer, jintArray tokenList, jint pos,
		jint size, jboolean removeSpecial, jboolean unparseSpecial) {
	auto *model = argeo::jni::as_pointer<llama_model*>(pointer);

//	llama_token tokens[length];
//	env->GetIntArrayRegion(tokenList, offset, length,
//			reinterpret_cast<jint*>(tokens));
//	jboolean *is_copy;
	void *tokens_arr = env->GetPrimitiveArrayCritical(tokenList,
	NULL);
	llama_token *tokens = static_cast<llama_token*>(tokens_arr) + pos;

	std::string text = jjml_tokens_to_cpp_string(model, tokens, size,
			removeSpecial, unparseSpecial);

	// clean up
	env->ReleasePrimitiveArrayCritical(tokenList, tokens_arr, 0);

	jbyteArray res = env->NewByteArray(text.size());
	env->SetByteArrayRegion(res, 0, text.size(),
			reinterpret_cast<jbyte*>(text.data()));
	return res;
}

JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeAsUtf8Bytes(
		JNIEnv *env, jclass, jlong pointer, jobject tokensBuf, jint pos,
		jint size, jboolean removeSpecial, jboolean unparseSpecial) {
	auto *model = argeo::jni::as_pointer<llama_model*>(pointer);

	void *tokens_arr = env->GetDirectBufferAddress(tokensBuf);
	if (tokens_arr == NULL)
		throw std::invalid_argument("Output is not a direct buffer");
	llama_token *tokens = static_cast<llama_token*>(tokens_arr) + pos;

	std::string text = jjml_tokens_to_cpp_string(model, tokens, size,
			removeSpecial, unparseSpecial);

	jbyteArray res = env->NewByteArray(text.size());
	env->SetByteArrayRegion(res, 0, text.size(),
			reinterpret_cast<jbyte*>(text.data()));
	return res;
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeAsUtf8(
		JNIEnv *env, jclass, jlong pointer, jobject tokensBuf, jint pos,
		jint size, jobject u8Buf, jint offset, jint length,
		jboolean removeSpecial, jboolean unparseSpecial) {
	jint n_chars = 0;
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(pointer);

		// input
		void *tokens_arr = env->GetDirectBufferAddress(tokensBuf);
		if (tokens_arr == NULL)
			throw std::invalid_argument("Output is not a direct buffer");
		llama_token *tokens = static_cast<llama_token*>(tokens_arr) + pos;

		// output
		void *u8_arr = env->GetDirectBufferAddress(u8Buf);
		if (u8_arr == NULL)
			throw std::invalid_argument("Input is not a direct buffer");
		assert(env->GetDirectBufferCapacity(u8Buf) >= offset + length);
		char *u8_chars = static_cast<char*>(u8_arr) + offset;

		n_chars = llama_detokenize(model, tokens, size, u8_chars, length,
				removeSpecial, unparseSpecial);

//		if (n_chars < 0)
//			throw std::range_error(
//					"Output buffer capacity " + std::to_string(length)
//							+ " is too small for " + std::to_string(-n_chars)
//							+ " characters - " + __func__);
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
	}
	return n_chars;
}

JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doTokenizeStringAsArray(
		JNIEnv *env, jclass, jlong pointer, jstring str, jboolean addSpecial,
		jboolean parseSpecial) {
	auto *model = argeo::jni::as_pointer<llama_model*>(pointer);

	const jchar *jchars = env->GetStringCritical(str, nullptr);
	jsize length = env->GetStringLength(str);
	std::u16string u16text = argeo::jni::jchars_to_utf16(jchars, length);
	std::string text = utf16_converter.to_bytes(u16text);

	std::vector<llama_token> tokens = jjml_cpp_string_to_tokens(model,
			text.data(), text.length(), addSpecial, parseSpecial);

	// clean up
	env->ReleaseStringCritical(str, jchars);

	jintArray res = env->NewIntArray(tokens.size());
	env->SetIntArrayRegion(res, 0, tokens.size(),
			reinterpret_cast<jint*>(tokens.data()));
	return res;
}

JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeArrayAsString(
		JNIEnv *env, jclass, jlong pointer, jintArray tokenList, jint pos,
		jint size, jboolean removeSpecial, jboolean unparseSpecial) {
	auto *model = argeo::jni::as_pointer<llama_model*>(pointer);

	// input
	void *tokens_arr = env->GetPrimitiveArrayCritical(tokenList,
	NULL);
	llama_token *tokens = static_cast<llama_token*>(tokens_arr) + pos;

	std::string text = jjml_tokens_to_cpp_string(model, tokens, size,
			removeSpecial, unparseSpecial);

	// clean up
	env->ReleasePrimitiveArrayCritical(tokenList, tokens_arr, 0);

	std::u16string u16text = utf16_converter.from_bytes(text);
	return argeo::jni::utf16_to_jstring(env, u16text);
}

JNIEXPORT jstring JNICALL Java_org_argeo_jjml_llama_LlamaCppVocabulary_doDeTokenizeAsString(
		JNIEnv *env, jclass, jlong pointer, jobject tokensBuf, jint pos,
		jint size, jboolean removeSpecial, jboolean unparseSpecial) {
	auto *model = argeo::jni::as_pointer<llama_model*>(pointer);

	void *tokens_arr = env->GetDirectBufferAddress(tokensBuf);
	llama_token *tokens = static_cast<llama_token*>(tokens_arr) + pos;

	std::string text = jjml_tokens_to_cpp_string(model, tokens, size,
			removeSpecial, unparseSpecial);

	std::u16string u16text = utf16_converter.from_bytes(text);
	return argeo::jni::utf16_to_jstring(env, u16text);
}

