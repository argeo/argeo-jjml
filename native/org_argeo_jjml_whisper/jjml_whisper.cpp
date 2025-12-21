#include <string>

#include <argeo/jni/argeo_jni.h>

#include <whisper.h>

#include "org_argeo_jjml_whisper_WhisperCppContext.h" // IWYU pragma: keep
#include "org_argeo_jjml_whisper_WhisperCppProcessor.h" // IWYU pragma: keep

/*
 * PROCESSOR
 */
JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_whisper_WhisperCppProcessor_doFull(
		JNIEnv *env, jclass, jlong contextPointer, jobject inputBuf,
		jint offset, jint length) {
	auto *ctx = argeo::jni::as_pointer<whisper_context*>(contextPointer);

	whisper_full_params wparams = whisper_full_default_params(
			WHISPER_SAMPLING_GREEDY);
	wparams.single_segment = true;
	wparams.translate = false;
//	wparams.language = "fr";

	float *pcmf32 = static_cast<float*>(env->GetDirectBufferAddress(inputBuf));
	if (whisper_full(ctx, wparams, pcmf32 + offset, length) != 0) {
		// TODO throw exception
	}

	jbyteArray res = nullptr;
	const int n_segments = whisper_full_n_segments(ctx);
	for (int i = 0; i < n_segments; ++i) {
		const char *text = whisper_full_get_segment_text(ctx, i);
		std::string str(text);
		res = env->NewByteArray(str.length());
		jbyte *bytes = env->GetByteArrayElements(res, 0);
		for (int i = 0; i < str.length(); i++)
			bytes[i] = text[i];
		env->ReleaseByteArrayElements(res, bytes, 0);
		//memcpy(bytes, str.c_str(), str.length());
		//std::copy(std::begin(bytes), std::end(str.c_str()), std::begin(str.c_str()));
		const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
		const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
	}
	return res;
}
/*
 * CONTEXT
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_whisper_WhisperCppContext_doInit(
		JNIEnv *env, jclass, jbyteArray path, jboolean useGpu,
		jboolean flashAttention) {
	try {
		struct whisper_context_params cparams =
				whisper_context_default_params();

		cparams.use_gpu = useGpu;
		cparams.flash_attn = flashAttention;

		std::string model_path = argeo::jni::to_string(env, path);

		struct whisper_context *ctx = whisper_init_from_file_with_params(
				model_path.c_str(), cparams);
		return (jlong) ctx;
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return 0;
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_whisper_WhisperCppContext_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *ctx = argeo::jni::as_pointer<whisper_context*>(env, obj);
	whisper_free(ctx);
}
