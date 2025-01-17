#include <cmath>

#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "jjml_llama.h"
#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppEmbeddingProcessor.h" // IWYU pragma: keep

/*
 * EMBEDDING
 */
// from llama.cpp's common llama_embd_normalize
static void embd_normalize(const float *inp, float *out, int n, int embd_norm) {
	double sum = 0.0;

	switch (embd_norm) {
	case -1: // no normalisation
		sum = 1.0;
		break;
	case 0: // max absolute
		for (int i = 0; i < n; i++) {
			if (sum < std::abs(inp[i]))
				sum = std::abs(inp[i]);
		}
		sum /= 32760.0; // make an int16 range
		break;
	case 2: // euclidean
		for (int i = 0; i < n; i++) {
			sum += inp[i] * inp[i];
		}
		sum = std::sqrt(sum);
		break;
	default: // p-norm (euclidean is p-norm p=2)
		for (int i = 0; i < n; i++) {
			sum += std::pow(std::abs(inp[i]), embd_norm);
		}
		sum = std::pow(sum, 1.0 / embd_norm);
		break;
	}

	const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

	for (int i = 0; i < n; i++) {
		out[i] = inp[i] * norm;
	}
}

// from llama.cpp's example/embedding
static void embd_batch_decode(llama_context *ctx, llama_batch &batch,
		float *output, int n_seq, int n_embd, int embd_norm) {
	const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
	const struct llama_model *model = llama_get_model(ctx);

	// clear previous kv_cache values (irrelevant for embeddings)
	llama_kv_cache_clear(ctx);

	// run model
//    LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
	if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
		// encoder-only model
		if (llama_encode(ctx, batch) < 0) {
//            LOG_ERR("%s : failed to encode\n", __func__);
		}
	} else if (!llama_model_has_encoder(model)
			&& llama_model_has_decoder(model)) {
		// decoder-only model
		if (llama_decode(ctx, batch) < 0) {
//            LOG_ERR("%s : failed to decode\n", __func__);
		}
	}

	for (int i = 0; i < batch.n_tokens; i++) {
		if (!batch.logits[i]) {
			continue;
		}

		const float *embd = nullptr;
		int embd_pos = 0;

		if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
			// try to get token embeddings
			embd = llama_get_embeddings_ith(ctx, i);
			embd_pos = i;
			GGML_ASSERT(embd != NULL && "failed to get token embeddings");
		} else {
			// try to get sequence embeddings - supported only when pooling_type is not NONE
			embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
			embd_pos = batch.seq_id[i][0];
			GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
		}

		float *out = output + embd_pos * n_embd;
		embd_normalize(embd, out, n_embd, embd_norm);
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppEmbeddingProcessor_doProcessEmbeddings(
		JNIEnv *env, jclass, jlong contextPointer, jobjectArray tokenLists,
		jfloatArray res) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);

	// TODO deal with normalization
	int embd_normalize = -1;

	int n_embd = llama_model_n_embd(llama_get_model(ctx));
	int n_batch = llama_n_batch(ctx);
	const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

	struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

	int n_prompts = env->GetArrayLength(tokenLists);

	jfloat *emb = (jfloat*) env->GetPrimitiveArrayCritical(res, nullptr);

	// break into batches
	int e = 0; // number of embeddings already stored
	int s = 0; // number of prompts in current batch
	for (int k = 0; k < n_prompts; k++) {
		jintArray tokenList = (jintArray) env->GetObjectArrayElement(tokenLists,
				k);

		const uint64_t n_toks = env->GetArrayLength(tokenList);

		// encode if at capacity
		if (batch.n_tokens + n_toks > n_batch) {
			float *out = emb + e * n_embd;
			embd_batch_decode(ctx, batch, out, s, n_embd, embd_normalize);
			e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
			s = 0;
			jjml_llama_batch_clear(batch);
		}

		// add to batch
//		embd_batch_add_seq(batch, inp, s);
		size_t n_tokens = env->GetArrayLength(tokenList);
		int *tokens = (int*) env->GetPrimitiveArrayCritical(tokenList, nullptr);
		for (size_t i = 0; i < n_tokens; i++) {
			jjml_llama_batch_add(batch, tokens[i], i, { s }, true);
		}
		env->ReleasePrimitiveArrayCritical(tokenList, tokens, 0);

		s += 1;
	}

	// final batch
	float *out = emb + e * n_embd;
	embd_batch_decode(ctx, batch, out, s, n_embd, embd_normalize);

	env->ReleasePrimitiveArrayCritical(res, emb, 0);
}

