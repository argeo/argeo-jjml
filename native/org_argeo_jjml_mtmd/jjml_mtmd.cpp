#include <string>
#include <vector>
#include <iostream>

#include <argeo/jni/argeo_jni.h>

#include <mtmd.h>

#include "org_argeo_jjml_mtmd_MtmdBackend.h" // IWYU pragma: keep
#include "org_argeo_jjml_mtmd_MtmdContext.h" // IWYU pragma: keep
#include "org_argeo_jjml_mtmd_MtmdProcessor.h" // IWYU pragma: keep
#include "org_argeo_jjml_mtmd_MtmdBitmap.h" // IWYU pragma: keep
#include "org_argeo_jjml_mtmd_MtmdImageBitmap.h" // IWYU pragma: keep

#include "jjml_mtmd.h"

/*
 * PROCESSOR
 */
JNIEXPORT jintArray JNICALL Java_org_argeo_jjml_mtmd_MtmdProcessor_doSingleTurn(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerChainPointer,
		jlong mtmdContextPointer, jbyteArray promptStr,
		jobjectArray bitmapsArr) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);
	const llama_model *model = llama_get_model(ctx);
	const llama_vocab *vocab = llama_model_get_vocab(model);
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(samplerChainPointer);
	auto *mtmd_ctx = argeo::jni::as_pointer<mtmd_context*>(mtmdContextPointer);

	std::string prompt = argeo::jni::to_string(env, promptStr);
//	std::cout << prompt << std::endl;

	mtmd_input_text text;
	text.text = prompt.c_str();
	text.add_special = true;
	text.parse_special = true;

//	std::cout << text.text << std::endl;

	size_t n_bitmaps = env->GetArrayLength(bitmapsArr);
	std::vector<const mtmd_bitmap*> bitmaps(n_bitmaps);
	for (int i = 0; i < n_bitmaps; i++) {
		jobject bitmapObj = env->GetObjectArrayElement(bitmapsArr, i);
		mtmd_bitmap *bitmap = argeo::jni::as_pointer<mtmd_bitmap*>(env,
				bitmapObj);
		bitmaps[i] = bitmap;
	}

	// Tokenize

	//std::cout << "# MTMD - Tokenize" << std::endl;
	mtmd_input_chunks *input_chunks = mtmd_input_chunks_init();
	const mtmd_bitmap **bitmaps_data = bitmaps.data();
	int tokenize_res = mtmd_tokenize(mtmd_ctx, input_chunks, &text,
			bitmaps_data, n_bitmaps);
	if (tokenize_res != 0)
		return nullptr; // FIXME throw exception

	// Evaluate

	//std::cout << "# MTMD - Evaluate" << std::endl;
	// FIXME deal with position properly
	llama_pos n_past = 0;
	// FIXME get n_batch from parameters
	int32_t n_batch = 256;

	llama_pos new_n_past;
	if (jjml_mtmd_eval_chunks(mtmd_ctx, ctx, // lctx
			input_chunks, // chunks
			n_past, // n_past
			0, // seq_id
			n_batch, // n_batch
			true, // logits_last
			&new_n_past)) {
		return nullptr; // TOD throw exception
	}

	n_past = new_n_past;

	// Generate response
	//std::cout << "# MTMD - Generate Response" << std::endl;
	llama_batch batch = llama_batch_init(1, 0, 1);
	int n_predict = 1000; // FIXME
	std::vector<llama_token> generated_tokens;
	for (int i = 0; i < n_predict; i++) {
		if (i > n_predict) {
			break;
		}

		llama_token token_id = llama_sampler_sample(smpl, ctx, -1);
		generated_tokens.push_back(token_id);
//        common_sampler_accept(ctx.smpl, token_id, true);

		if (llama_vocab_is_eog(vocab, token_id)) {
			break; // end of generation
		}

//		LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
//		fflush(stdout);

		// eval the token
		jjml_mtmd_batch_clear(batch);
		jjml_mtmd_batch_add(batch, token_id, n_past++, { 0 }, true);
		if (llama_decode(ctx, batch)) {
			// TODO throw exception
		}
	}

	jintArray res = nullptr;
	res = env->NewIntArray(generated_tokens.size());
	jint *bytes = env->GetIntArrayElements(res, 0);
	for (int i = 0; i < generated_tokens.size(); i++)
		bytes[i] = generated_tokens[i];
	env->ReleaseIntArrayElements(res, bytes, 0);

	return res;
}

/*
 * CONTEXT
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_mtmd_MtmdContext_doInit(JNIEnv *env,
		jclass, jobject modelObj, jbyteArray path, jboolean useGpu,
		jint threads) {
	try {
		auto *model = argeo::jni::as_pointer<llama_model*>(env, modelObj);

		mtmd::context_ptr ctx_vision;
		std::string mmproj_path = argeo::jni::to_string(env, path);
		mtmd_context_params mparams = mtmd_context_params_default();
		mparams.use_gpu = useGpu;
		mparams.print_timings = true;
		mparams.n_threads = threads;
		//mparams.verbosity = GGML_LOG_LEVEL_DEBUG; // GGML_LOG_LEVEL_INFO;
		mtmd_context *mtmd_ctx = mtmd_init_from_file(mmproj_path.c_str(), model,
				mparams);
		return (jlong) mtmd_ctx;
	} catch (const std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
		return 0;
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_mtmd_MtmdContext_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *mtmd_ctx = argeo::jni::as_pointer<mtmd_context*>(env, obj);
	mtmd_free(mtmd_ctx);
}

/*
 * BITMAP
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_mtmd_MtmdImageBitmap_doInit(
		JNIEnv *env, jclass, jobject inputBuf, jint width, jint height) {
	const unsigned char *data =
			static_cast<const unsigned char*>(env->GetDirectBufferAddress(
					inputBuf));
	mtmd_bitmap *bitmap = mtmd_bitmap_init(width, height, data);
	return (jlong) bitmap;
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_mtmd_MtmdImageBitmap_doInitFromBytes(
		JNIEnv *env, jclass, jbyteArray bytes, jint offset, jint width,
		jint height) {
	void *arr = env->GetPrimitiveArrayCritical(bytes, 0);
	const unsigned char *data = static_cast<const unsigned char*>(arr) + offset;
	mtmd_bitmap *bitmap = mtmd_bitmap_init(width, height, data);

	// clean up
	env->ReleasePrimitiveArrayCritical(bytes, arr, 0);

	return (jlong) bitmap;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_mtmd_MtmdBitmap_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *bitmap = argeo::jni::as_pointer<mtmd_bitmap*>(env, obj);
	mtmd_bitmap_free(bitmap);
}

/*
 * BACKEND
 */
JNIEXPORT jbyteArray JNICALL Java_org_argeo_jjml_mtmd_MtmdBackend_doGetDefaultMarker(
		JNIEnv *env, jclass) {
	// TODO Factorize
	const char *text = mtmd_default_marker();
	std::string str(text);
	jbyteArray res = env->NewByteArray(str.length());
	jbyte *bytes = env->GetByteArrayElements(res, 0);
	for (int i = 0; i < str.length(); i++)
		bytes[i] = text[i];
	env->ReleaseByteArrayElements(res, bytes, 0);
	return res;
}
