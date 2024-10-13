#include <argeo/argeo_jni.h>
#include <ggml.h>
#include <jni.h>
#include <jni_md.h>
#include <llama.h>
#include <stddef.h>
#include <algorithm>
#include <iostream>
#include <vector>

#include "jjml_llama.h"
#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppBatchProcessor.h" // IWYU pragma: keep

/*
 * LLAMA UTILITIES
 */

/*
 * BATCH
 */

static void jjml_evaluate_inital_tokens(JNIEnv *env, llama_context *ctx,
		llama_batch &batch, std::vector<llama_seq_id> &seq_ids) {
	const llama_model *model = llama_get_model(ctx);
	if (llama_model_has_encoder(model)) {
		if (llama_encode(ctx, batch)) {
			env->ThrowNew(IllegalStateException, "Failed to encode");
			return;
		}

		llama_token decoder_start_token_id = llama_model_decoder_start_token(
				model);
		if (decoder_start_token_id == -1) {
			decoder_start_token_id = llama_token_bos(model);
		}

		jjml_llama_batch_clear(batch);
		jjml_llama_batch_add(batch, decoder_start_token_id, 0, seq_ids, false);
	}

	// llama_decode will output logits only for the last token of the prompt
	batch.logits[batch.n_tokens - 1] = true;

	if (llama_decode(ctx, batch) != 0) {
		env->ThrowNew(IllegalStateException, "Decode failed");
	}

}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doProcessSingleBatch(
		JNIEnv *env, jobject, jlong contextPointer, jobject buf,
		jint n_predict) {
	auto *ctx = argeo::jni::getPointer<llama_context*>(contextPointer);
	const llama_model *model = llama_get_model(ctx);

	const int n_parallel = 1;

	void *bufRegion = env->GetDirectBufferAddress(buf);
	int bufCapacity = env->GetDirectBufferCapacity(buf);

	jclass ByteBuffer = env->FindClass("java/nio/ByteBuffer");
	jmethodID ByteBuffer$limit = env->GetMethodID(ByteBuffer, "limit", "()I");
	jmethodID ByteBuffer$setLimit = env->GetMethodID(ByteBuffer, "limit",
			"(I)Ljava/nio/ByteBuffer;");
	jmethodID ByteBuffer$setPosition = env->GetMethodID(ByteBuffer, "position",
			"(I)Ljava/nio/ByteBuffer;");
	int bufLimit = env->CallIntMethod(buf, ByteBuffer$limit);

	// TODO check modulo
	int system_prompt_tokens_size = bufLimit / sizeof(llama_token);
	llama_token *tokens = static_cast<llama_token*>(bufRegion);

	llama_batch batch = llama_batch_init(
			std::max((size_t) system_prompt_tokens_size, (size_t) n_parallel),
			0, n_parallel);

	std::vector<llama_seq_id> seq_ids(n_parallel, 0);
	seq_ids[0] = 0;

	// evaluate the initial prompt
	// TODO memcopy?
	for (size_t i = 0; i < system_prompt_tokens_size; i++) {
		jjml_llama_batch_add(batch, tokens[i], i, seq_ids, false);
	}
	GGML_ASSERT(batch.n_tokens == (int ) system_prompt_tokens_size);

	//std::cerr << "Position: " << batch.n_tokens << std::endl;

	jjml_evaluate_inital_tokens(env, ctx, batch, seq_ids);

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

	int n_cur = batch.n_tokens;
	int n_decode = 0;

	const auto t_main_start = ggml_time_us();

	int next_token_pos = batch.n_tokens;
	int start_generation_pos = next_token_pos;
	while (n_cur <= n_predict) {
		// prepare the next batch
		jjml_llama_batch_clear(batch);

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
				result[i].clear();
				continue;
			}

			//streams[i] += llama_token_to_piece(ctx, new_token_id);
			result[i].push_back(new_token_id);

			tokens[next_token_pos] = new_token_id;
			next_token_pos++;
			std::cerr << next_token_pos << "\t" << new_token_id << std::endl;

			i_batch[i] = batch.n_tokens;

			// push this new token for next evaluation
			jjml_llama_batch_add(batch, new_token_id, n_cur, { i }, true);

			n_decode += 1;
		}

		// all streams are finished
		if (batch.n_tokens == 0) {
			break;
		}

		n_cur += 1;

		// evaluate the current batch with the transformer model
		if (llama_decode(ctx, batch)) {
			env->ThrowNew(IllegalStateException, "Failed to decode");
		}
	}

	env->CallVoidMethod(buf, ByteBuffer$setLimit,
			next_token_pos * sizeof(llama_token));
	env->CallVoidMethod(buf, ByteBuffer$setPosition,
			start_generation_pos * sizeof(llama_token));
	return 0;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doProcessBatch(
		JNIEnv *env, jobject, jobjectArray callbacks, jobject contextObj,
		jintArray systemPromptTokens, jobjectArray, jint n_predict) {
	auto *ctx = getPointer<llama_context*>(env, contextObj);
	const int n_ctx = llama_n_ctx(ctx);

	const llama_model *model = llama_get_model(ctx);

	const int n_parallel = env->GetArrayLength(callbacks);

	//std::vector<llama_token> tokens_list;

	jsize system_prompt_tokens_size = env->GetArrayLength(systemPromptTokens);
	jboolean isCopy;
	llama_token *system_prompt_tokens =
			(llama_token*) env->GetPrimitiveArrayCritical(systemPromptTokens,
					&isCopy);
//	jint *system_prompt_tokens = (jint*) env->GetIntArrayElements(
//			systemPromptTokens, &isCopy);

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
		jjml_llama_batch_add(batch, system_prompt_tokens[i], i, seq_ids, false);
	}
	GGML_ASSERT(batch.n_tokens == (int ) system_prompt_tokens_size);

	// array is copied, we can release it
	env->ReleasePrimitiveArrayCritical(systemPromptTokens, system_prompt_tokens,
			0);
//	env->ReleaseIntArrayElements(systemPromptTokens, system_prompt_tokens,
//			0);

//	if (llama_model_has_encoder(model)) {
//		if (llama_encode(ctx, batch)) {
//			env->ThrowNew(IllegalStateException, "Failed to encode");
//			return;
//		}
//
//		llama_token decoder_start_token_id = llama_model_decoder_start_token(
//				model);
//		if (decoder_start_token_id == -1) {
//			decoder_start_token_id = llama_token_bos(model);
//		}
//
//		jjml_llama_batch_clear(batch);
//		jjml_llama_batch_add(batch, decoder_start_token_id, 0, seq_ids, false);
//	}
//
//	// llama_decode will output logits only for the last token of the prompt
//	batch.logits[batch.n_tokens - 1] = true;
//
//	if (llama_decode(ctx, batch) != 0) {
//		env->ThrowNew(IllegalStateException, "Decode failed");
//	}
	jjml_evaluate_inital_tokens(env, ctx, batch, seq_ids);

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

	int n_cur = batch.n_tokens;
	int n_decode = 0;

	const auto t_main_start = ggml_time_us();

	while (n_cur <= n_predict) {
		// prepare the next batch
		jjml_llama_batch_clear(batch);

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
			jjml_llama_batch_add(batch, new_token_id, n_cur, { i }, true);

			n_decode += 1;
		}

		// all streams are finished
		if (batch.n_tokens == 0) {
			break;
		}

		n_cur += 1;

		// evaluate the current batch with the transformer model
		if (llama_decode(ctx, batch)) {
			env->ThrowNew(IllegalStateException, "Failed to decode");
		}
	}

}
