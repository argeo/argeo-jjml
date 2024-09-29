#include <iostream>
#include <vector>
#include <algorithm>

#include <llama.h>

#include "org_argeo_jjml_llama_LlamaCppBatchProcessor.h"
#include "org_argeo_jjml_llama_LlamaCppContext.h"
#include "org_argeo_jjml_llama_LlamaCppModel.h"
#include "org_argeo_jjml_llama_LlamaCppBackend.h"

/**
 * JNI Utilities
 */
jlong getPointer(JNIEnv *env, jobject obj) {
	jclass clss = env->GetObjectClass(obj);
	jmethodID method = env->GetMethodID(clss, "getPointer", "()J");
	// TODO check that method was found
	jlong pointer = env->CallLongMethod(obj, method);
	return pointer;
}

void ThrowIllegalStateException(JNIEnv *env, char const *message) {
	jclass IllegalStateException = env->FindClass(
			"java/lang/IllegalStateException");
	env->ThrowNew(IllegalStateException, message);
}

/*
 * Lllama Utilities
 */

void llama_batch_add(struct llama_batch &batch, llama_token id, llama_pos pos,
		const std::vector<llama_seq_id> &seq_ids, bool logits) {
	batch.token[batch.n_tokens] = id;
	batch.pos[batch.n_tokens] = pos;
	batch.n_seq_id[batch.n_tokens] = seq_ids.size();
	for (size_t i = 0; i < seq_ids.size(); ++i) {
		batch.seq_id[batch.n_tokens][i] = seq_ids[i];
	}
	batch.logits[batch.n_tokens] = logits;

	batch.n_tokens++;
}

void llama_batch_clear(struct llama_batch &batch) {
	batch.n_tokens = 0;
}

/*
 * Batch processor
 */
JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doProcessBatch(
		JNIEnv *env, jobject, jobjectArray callbacks, jobject contextObj,
		jintArray systemPromptTokens, jobjectArray, jint n_predict) {
	llama_context *ctx = (llama_context*) getPointer(env, contextObj);
	const int n_ctx = llama_n_ctx(ctx);

	const llama_model *model = llama_get_model(ctx);

	// sampling
	auto sparams = llama_sampler_chain_default_params();

	llama_sampler *smpl = llama_sampler_chain_init(sparams);

	const int top_k = 40;
	const float top_p = 0.9f;
	const float min_p = 0.9f;
	const float temp = 0.4f;
	llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
	llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, min_p));
	llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
	// !! crashes if seed is not set
	llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

	const int n_parallel = env->GetArrayLength(callbacks);

	//std::vector<llama_token> tokens_list;

	jsize system_prompt_tokens_size = env->GetArrayLength(systemPromptTokens);
	jboolean *is_copy;
	jint *system_prompt_tokens = (jint*) env->GetPrimitiveArrayCritical(
			systemPromptTokens, is_copy);

	// make sure the KV cache is big enough to hold all the prompt and generated tokens
//	if (n_kv_req > n_ctx) {
//		LOG_TEE(
//				"%s: error: n_kv_req (%d) > n_ctx, the required KV cache size is not big enough\n",
//				__func__, n_kv_req);
//		LOG_TEE("%s:        either reduce n_parallel or increase n_ctx\n",
//				__func__);
//		return 1;
//	}

// create a llama_batch
// we use this object to submit token data for decoding
	llama_batch batch = llama_batch_init(
			std::max((size_t) system_prompt_tokens_size, (size_t) n_parallel),
			0, n_parallel);

	std::vector<llama_seq_id> seq_ids(n_parallel, 0);
	for (int32_t i = 0; i < n_parallel; ++i) {
		seq_ids[i] = i;
	}

	// evaluate the initial prompt
	// TODO memcopy?
	for (size_t i = 0; i < system_prompt_tokens_size; i++) {
		llama_batch_add(batch, system_prompt_tokens[i], i, seq_ids, false);
	}
	GGML_ASSERT(batch.n_tokens == (int ) system_prompt_tokens_size);

	// array is released, we can copy it
	env->ReleasePrimitiveArrayCritical(systemPromptTokens, system_prompt_tokens,
			0);

	if (llama_model_has_encoder(model)) {
		if (llama_encode(ctx, batch)) {
			ThrowIllegalStateException(env, "Failed to encode");
		}

		llama_token decoder_start_token_id = llama_model_decoder_start_token(
				model);
		if (decoder_start_token_id == -1) {
			decoder_start_token_id = llama_token_bos(model);
		}

		llama_batch_clear(batch);
		llama_batch_add(batch, decoder_start_token_id, 0, seq_ids, false);
	}

	// llama_decode will output logits only for the last token of the prompt
	batch.logits[batch.n_tokens - 1] = true;

	if (llama_decode(ctx, batch) != 0) {
		ThrowIllegalStateException(env, "Decode failed");
	}

	// main loop

	// we will store the parallel decoded sequences in this vector
	//std::vector<std::string> streams(n_parallel);

	// TODO optimize capacity allocation
	std::vector<std::vector<llama_token>> result(n_parallel);
	for (int32_t i = 0; i < n_parallel; ++i) {
//		std::vector<llama_token> *res = new std::vector<llama_token>();
//		result.push_back(*res);

		result[i] = std::vector<llama_token>();
	}

	// remember the batch index of the last token for each parallel sequence
	// we need this to determine which logits to sample from
	std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

	int n_cur = batch.n_tokens;
	int n_decode = 0;

	const auto t_main_start = ggml_time_us();

	while (n_cur <= n_predict) {
		// prepare the next batch
		llama_batch_clear(batch);

		// sample the next token for each parallel sequence / stream
		for (int32_t i = 0; i < n_parallel; ++i) {
			if (i_batch[i] < 0) {
				// the stream has already finished
				continue;
			}

			const llama_token new_token_id = llama_sampler_sample(smpl, ctx,
					i_batch[i]);

			//const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

			// is it an end of generation? -> mark the stream as finished
			if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict) {
				i_batch[i] = -1;
				jobject callback = env->GetObjectArrayElement(callbacks, i);
				jclass CompletableFuture = env->GetObjectClass(callback);
				jmethodID completeMethod = env->GetMethodID(CompletableFuture,
						"complete", "(Ljava/lang/Object;)Z");

				jintArray resTokens = env->NewIntArray(result[i].size());
				env->SetIntArrayRegion(resTokens, 0, result[i].size(),
						result[i].data());

				env->CallVoidMethod(callback, completeMethod, resTokens);

				// clear data
				// TODO create each vector with new, then delete
//				delete &result[i];
				result[i].clear();
				continue;
			}

			//streams[i] += llama_token_to_piece(ctx, new_token_id);
			result[i].push_back(new_token_id);

			i_batch[i] = batch.n_tokens;

			// push this new token for next evaluation
			llama_batch_add(batch, new_token_id, n_cur, { i }, true);

			n_decode += 1;
		}

		// all streams are finished
		if (batch.n_tokens == 0) {
			break;
		}

		n_cur += 1;

		// evaluate the current batch with the transformer model
		if (llama_decode(ctx, batch)) {
			ThrowIllegalStateException(env, "Failed to decode");
		}
	}

}

/*
 * Llama context
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doInit(
		JNIEnv *env, jobject obj, jobject modelObj) {
	llama_model *model = (llama_model*) getPointer(env, modelObj);

	llama_context_params ctx_params = llama_context_default_params();

	llama_context *ctx = llama_new_context_with_model(model, ctx_params);

	if (ctx == NULL) {
		ThrowIllegalStateException(env, "Failed to create llama.cpp context");
	}
	return (jlong) ctx;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppContext_doDestroy(
		JNIEnv *env, jobject obj) {
	llama_context *ctx = (llama_context*) getPointer(env, obj);
	llama_free(ctx);
}

/*
 * Llama model
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doInit(
		JNIEnv *env, jobject obj, jstring localPath) {
	const char *path_model = env->GetStringUTFChars(localPath, NULL);

	llama_model_params mparams = llama_model_default_params();
	//mparams.n_gpu_layers = 33;

	llama_model *model = llama_load_model_from_file(path_model, mparams);

	env->ReleaseStringUTFChars(localPath, path_model);
	return (jlong) model;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppModel_doDestroy(
		JNIEnv *env, jobject obj) {
	llama_model *model = (llama_model*) getPointer(env, obj);
	llama_free_model(model);
}

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
 * Llama backend
 */
JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_doInit(JNIEnv*,
		jobject) {
	llama_backend_init();
	//llama_numa_init(params.numa);
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_doDestroy(
		JNIEnv*, jobject) {
	llama_backend_free();
}

