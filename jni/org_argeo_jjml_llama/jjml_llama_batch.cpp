#include <argeo/argeo_jni.h>
#include <bits/stdint-intn.h>
#include <ggml.h>
#include <jni.h>
#include <jni_md.h>
#include <llama.h>
#include <stddef.h>
#include <algorithm>
#include <vector>
#include <cassert>

#include "jjml_llama.h"
#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppBatchProcessor.h" // IWYU pragma: keep

/*
 * LLAMA UTILITIES
 */

/*
 * BATCH
 */
static void jjml_populate_default_samplers(const llama_model *model,
		llama_sampler *chain, //
		float temp = 0.80f, // <= 0.0 to sample greedily, 0.0 to not output probabilities
		int32_t top_k = 40,    // <= 0 to use vocab size
		float top_p = 0.95f, // 1.0 = disabled
		float min_p = 0.05f, // 0.0 = disabled
		float tfs_z = 1.00f, // 1.0 = disabled
		float typ_p = 1.00f, // typical_p, 1.0 = disabled
		float dynatemp_range = 0.00f, // 0.0 = disabled
		float dynatemp_exponent = 1.00f, // controls how entropy maps to temperature in dynamic temperature sampler
		int32_t penalty_last_n = 64, // last n tokens to penalize (0 = disable penalty, -1 = context size)
		float penalty_repeat = 1.00f, // 1.0 = disabled
		float penalty_freq = 0.00f, // 0.0 = disabled
		float penalty_present = 0.00f, // 0.0 = disabled
		bool penalize_nl = false, // consider newlines as a repeatable token
		bool ignore_eos = false, int32_t mirostat = 0, // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
		float mirostat_tau = 5.00f, // target entropy
		float mirostat_eta = 0.10f // learning rate
		) {
//    llama_sampler_chain_add(chain,
//            llama_sampler_init_logit_bias(
//                llama_n_vocab(model),
//                logit_bias.size(),
//                logit_bias.data())),

	llama_sampler_chain_add(chain,
			llama_sampler_init_penalties(llama_n_vocab(model),
					llama_token_eos(model), llama_token_nl(model),
					penalty_last_n, penalty_repeat, penalty_freq,
					penalty_present, penalize_nl, ignore_eos));
	if (temp > 0) {
		llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
		llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, min_p));
		llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));

		// final sampler
		llama_sampler_chain_add(chain,
				llama_sampler_init_dist(LLAMA_DEFAULT_SEED)); // !! crashes if seed is not set
	} else {
		llama_sampler_chain_add(chain, llama_sampler_init_greedy());
	}

}

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

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doWriteBatch(
		JNIEnv *env, jclass, jlong contextPointer, jint contextPosition,
		jobjectArray inputBuffers, jintArray sequenceIds, jboolean lastLogits) {
	auto *ctx = argeo::jni::getPointer<llama_context*>(contextPointer);
	const llama_model *model = llama_get_model(ctx);

	int cur_pos = contextPosition;

	const int n_parallel = env->GetArrayLength(sequenceIds);
	auto *sequence_ids = static_cast<llama_seq_id*>(env->GetIntArrayElements(
			sequenceIds, nullptr));

	int inputBuffersCount = env->GetArrayLength(inputBuffers);
	assert(inputBuffersCount > 0 && "Input buffers count");

	if (inputBuffersCount == 1) { // common prompt to all sequences

		jobject inputBuf = env->GetObjectArrayElement(inputBuffers, 0);

		PERF_BEGIN();

		int input_tokens_size = env->GetDirectBufferCapacity(inputBuf);
		llama_token *input_tokens =
				static_cast<llama_token*>(env->GetDirectBufferAddress(inputBuf));

		llama_batch batch = llama_batch_init(
				std::max((size_t) input_tokens_size, (size_t) n_parallel), 0,
				n_parallel);

		// evaluate the initial prompt
		// !! we use the Java direct buffer as source for the input tokens
//		free(batch.token);
//		free(batch.seq_id);
//		batch.token = input_tokens;
		for (size_t i = 0; i < input_tokens_size; i++) {
//			jjml_llama_batch_add(batch, input_tokens[i], i, seq_ids, false);
			batch.token[batch.n_tokens] = input_tokens[i];
			batch.pos[batch.n_tokens] = cur_pos + i;
			batch.n_seq_id[batch.n_tokens] = n_parallel;
			for (size_t i = 0; i < n_parallel; ++i) {
				batch.seq_id[batch.n_tokens][i] = sequence_ids[i];
			}
//			batch.seq_id[batch.n_tokens] = sequence_ids;
			batch.logits[batch.n_tokens] = false;

			batch.n_tokens++;

		}
		batch.n_tokens = input_tokens_size;
		GGML_ASSERT(batch.n_tokens == (int ) input_tokens_size);

		//std::cerr << "Position: " << batch.n_tokens << std::endl;

		// FIXME temporary backward compatibility
		std::vector<llama_seq_id> seq_ids(n_parallel, 0);
		for (int i = 0; i < n_parallel; i++) {
			seq_ids[i] = sequence_ids[i];
		}
		jjml_evaluate_inital_tokens(env, ctx, batch, seq_ids);

		cur_pos = cur_pos + batch.n_tokens;
//		curr_batch_pos = batch.pos[batch.n_tokens - 1];

		// !! dereference what Java owns before batch is freed
//		batch.token = nullptr;
//		for (size_t i = 0; i < input_tokens_size; i++) {
//			batch.seq_id[i] = nullptr;
//		}
		llama_batch_free(batch);

		PERF_END(__func__);
	} else {
		// TODO unsupported
	}

	// clean up
	env->ReleaseIntArrayElements(sequenceIds, sequence_ids, 0);

	return cur_pos;
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doReadBatch(
		JNIEnv *env, jclass, jlong contextPointer, jint contextPosition,
		jobjectArray outputBuffers, jintArray sequenceIds,
		jobject completionHandler) {
	auto *ctx = argeo::jni::getPointer<llama_context*>(contextPointer);
	const llama_model *model = llama_get_model(ctx);

	llama_pos cur_pos = static_cast<llama_pos>(contextPosition);

	const int n_parallel = env->GetArrayLength(sequenceIds);
	assert(n_parallel > 0 && "Sequence count");
//	auto *sequence_ids = static_cast<llama_seq_id*>(env->GetIntArrayElements(
//			sequenceIds, nullptr));
	jint *arr =env->GetIntArrayElements(sequenceIds, nullptr);
	llama_seq_id sequence_ids[n_parallel];
	for(int i=0;i<n_parallel;i++){
		sequence_ids[i] = static_cast<llama_seq_id>(arr[i]);
	}
	env->ReleaseIntArrayElements(sequenceIds, arr, 0);

	const int outBuffersCount = env->GetArrayLength(outputBuffers);
	assert(outBuffersCount == n_parallel && "As many buffers as sequences");

	llama_token *seq_tokens[n_parallel];
	int seq_tokens_size[n_parallel];
//	int total_output_capacity = 0;
	int max_decodes = 0;

	for (int i = 0; i < n_parallel; i++) {
		jobject outputBuf = env->GetObjectArrayElement(outputBuffers, i);
		void *output = env->GetDirectBufferAddress(outputBuf);
//		std::cerr << (jlong) output << std::endl;
		int outputCapacity = env->GetDirectBufferCapacity(outputBuf);
		assert(outputCapacity > 0 && "Output buffer capacity");

		int out_tokens_size = outputCapacity;
		llama_token *output_tokens = static_cast<llama_token*>(output);

		seq_tokens[i] = output_tokens;
		seq_tokens_size[i] = out_tokens_size;
		if (out_tokens_size > max_decodes)
			max_decodes = out_tokens_size;
	}

	PERF_BEGIN();
	// sampling
	auto sparams = llama_sampler_chain_default_params();
	llama_sampler *smpl = llama_sampler_chain_init(sparams);
	jjml_populate_default_samplers(model, smpl);
	//jjml_populate_default_samplers(model, smpl, 0); // deterministic

	// remember the batch index of the last token for each parallel sequence
	// we need this to determine which logits to sample from
	std::vector<int32_t> i_batch(n_parallel, cur_pos - 1);

	int next_idx = 0;

	llama_batch batch = llama_batch_init(n_parallel, 0, n_parallel);

	// FIXME deal more precisely with the upper limit
	//assert(max_decodes % sizeof(llama_token) == 0);
	int n_predict = cur_pos + max_decodes;
	while (cur_pos <= n_predict) {
		// prepare the next batch
		jjml_llama_batch_clear(batch);

		{
			// sample the next token for each parallel sequence / stream
			for (int32_t i = 0; i < n_parallel; ++i) {
				if (i_batch[i] < 0) {
					// the stream has already finished
					continue;
				}

				//PERF_BEGIN();
				const llama_token new_token_id = llama_sampler_sample(smpl, ctx,
						i_batch[i]);
				//PERF_END("sampling");

				// is it an end of generation? -> mark the stream as finished
				if (llama_token_is_eog(model, new_token_id) //
				|| next_idx == seq_tokens_size[i] //
				|| cur_pos == n_predict //
						) {
					i_batch[i] = -1;

					jobject outputBuf = env->GetObjectArrayElement(
							outputBuffers, i);
//					int byte_idx = next_idx; // * sizeof(llama_token);

					// set output buffer in a proper state
					env->CallVoidMethod(outputBuf, IntBuffer$limitI, next_idx);
					env->CallVoidMethod(outputBuf, IntBuffer$positionI,
							next_idx);

					// call completion handler
					jclass Integer = env->FindClass("java/lang/Integer");
					jobject completionHandlerResult =
							env->CallStaticObjectMethod(Integer,
									Integer$valueOf, next_idx);
					jobject completionHandlerAttachment =
							env->CallStaticObjectMethod(Integer,
									Integer$valueOf, i);
					env->CallVoidMethod(completionHandler,
							CompletionHandler$completed,
							completionHandlerResult,
							completionHandlerAttachment);
					continue;
				}

//				std::cerr << next_idx << "\t" << i << "\t" << new_token_id
//						<< std::endl;
				assert(next_idx < seq_tokens_size[i] && "No overflow");
				seq_tokens[i][next_idx] = new_token_id;

				i_batch[i] = batch.n_tokens;

				// push this new token for next evaluation
				jjml_llama_batch_add(batch, new_token_id, cur_pos, {
						sequence_ids[i] }, true);
			}
			next_idx++;
		}

		// all streams are finished
		if (batch.n_tokens == 0) {
			break;
		}

		cur_pos += 1;

		// evaluate the current batch with the transformer model
		if (llama_decode(ctx, batch)) {
			env->ThrowNew(IllegalStateException, "Failed to decode");
		}
	}

	// !! dereference what Java owns before batch is freed
//	for (size_t i = 0; i < n_parallel; i++) {
//		batch.seq_id[i] = nullptr;
//	}
	llama_batch_free(batch);
	PERF_END(__func__);

	// clean up
//	env->ReleaseIntArrayElements(sequenceIds, sequence_ids, 0);

	// TODO assert consistency of context position with regard to the output buffers sizes
	return cur_pos;
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doProcessSingleBatch(
		JNIEnv *env, jobject, jlong contextPointer, jobject inputBuf,
		jobject outputBuf) {
	auto *ctx = argeo::jni::getPointer<llama_context*>(contextPointer);
	const llama_model *model = llama_get_model(ctx);

	const int n_parallel = 1;
	const int sequenceId = 624;
	int cur_pos = 0;

	void *input = env->GetDirectBufferAddress(inputBuf);
	int inputCapacity = env->GetDirectBufferCapacity(inputBuf);
//	int inputLimit = env->CallIntMethod(inputBuf, ByteBuffer$limit);
//	assert(env->CallIntMethod(inputBuf, ByteBuffer$position) == 0);

//	llama_pos curr_batch_pos = 0;
	{ // hotspot
		PERF_BEGIN();

		assert(inputCapacity % sizeof(llama_token) == 0);
		int input_tokens_size = inputCapacity; // / sizeof(llama_token);
		llama_token *input_tokens = static_cast<llama_token*>(input);

		llama_batch batch = llama_batch_init(
				std::max((size_t) input_tokens_size, (size_t) n_parallel), 0,
				n_parallel);

		std::vector<llama_seq_id> seq_ids(n_parallel, 0);
		seq_ids[0] = sequenceId;

		// evaluate the initial prompt
		// TODO memcopy?
		batch.token = input_tokens;
		for (size_t i = 0; i < input_tokens_size; i++) {
//			jjml_llama_batch_add(batch, input_tokens[i], i, seq_ids, false);

//			batch.token[batch.n_tokens] = input_tokens[i];
			batch.pos[batch.n_tokens] = i;
			batch.n_seq_id[batch.n_tokens] = seq_ids.size();
//			for (size_t i = 0; i < seq_ids.size(); ++i) {
//				batch.seq_id[batch.n_tokens][i] = seq_ids[i];
//			}
			batch.seq_id[batch.n_tokens] = seq_ids.data();
			batch.logits[batch.n_tokens] = false;

			batch.n_tokens++;

		}
		batch.n_tokens = input_tokens_size;
		GGML_ASSERT(batch.n_tokens == (int ) input_tokens_size);

		//std::cerr << "Position: " << batch.n_tokens << std::endl;

		jjml_evaluate_inital_tokens(env, ctx, batch, seq_ids);

		cur_pos = cur_pos + batch.n_tokens;
//		curr_batch_pos = batch.pos[batch.n_tokens - 1];

		// dereference tokens since we own them
		batch.token = nullptr;
		for (size_t i = 0; i < input_tokens_size; i++) {
			batch.seq_id[i] = nullptr;
		}
		llama_batch_free(batch);

		PERF_END("jjml_evaluate_inital_tokens");
	}

	// output
	void *output = env->GetDirectBufferAddress(outputBuf);
	int outputCapacity = env->GetDirectBufferCapacity(outputBuf);
//	int outputLimit = env->CallIntMethod(outputBuf, ByteBuffer$limit);
//	assert(env->CallIntMethod(outputBuf, ByteBuffer$position) == 0);

	{
		PERF_BEGIN();
//		assert(outputCapacity % sizeof(llama_token) == 0);
//		int out_tokens_size = outputCapacity / sizeof(llama_token);
//		llama_token *output_tokens = static_cast<llama_token*>(output);
		int out_tokens_size = outputCapacity;
		llama_token *output_tokens = static_cast<llama_token*>(output);

		// sampling
		auto sparams = llama_sampler_chain_default_params();
		llama_sampler *smpl = llama_sampler_chain_init(sparams);
		jjml_populate_default_samplers(model, smpl);
		//jjml_populate_default_samplers(model, smpl, 0); // deterministic

		// remember the batch index of the last token for each parallel sequence
		// we need this to determine which logits to sample from
		std::vector<int32_t> i_batch(n_parallel, cur_pos - 1);

		int next_idx = 0;
		int n_decode = 0;

//	int start_generation_pos = next_idx;

		llama_batch batch = llama_batch_init(n_parallel, 0, n_parallel);
//	batch.pos[0] = curr_batch_pos; // not needed as it will be reset before being used

		assert(outputCapacity % sizeof(llama_token) == 0);
		int n_predict = cur_pos + outputCapacity / sizeof(llama_token);
		while (cur_pos <= n_predict) {

			// prepare the next batch
			jjml_llama_batch_clear(batch);

			{
				// sample the next token for each parallel sequence / stream
				for (int32_t i = 0; i < n_parallel; ++i) {
					if (i_batch[i] < 0) {
						// the stream has already finished
						continue;
					}

					//PERF_BEGIN();
					const llama_token new_token_id = llama_sampler_sample(smpl,
							ctx, i_batch[i]);
					//PERF_END("sampling");

					// is it an end of generation? -> mark the stream as finished
					if (llama_token_is_eog(model, new_token_id)
							|| cur_pos == n_predict) {
						i_batch[i] = -1;

						env->CallVoidMethod(outputBuf, IntBuffer$limitI,
								next_idx);
						env->CallVoidMethod(outputBuf, IntBuffer$positionI,
								next_idx);
//						env->CallVoidMethod(outputBuf, ByteBuffer$limitI,
//								next_idx * sizeof(llama_token));
//						env->CallVoidMethod(outputBuf, ByteBuffer$positionI,
//								next_idx * sizeof(llama_token));

//					result[i].clear();
						continue;
					}

//				result[i].push_back(new_token_id);

					//std::cerr << next_token_pos << "\t" << new_token_id << std::endl;
					output_tokens[next_idx] = new_token_id;
					next_idx++;

					i_batch[i] = batch.n_tokens;

					// push this new token for next evaluation
					jjml_llama_batch_add(batch, new_token_id, cur_pos, {
							sequenceId }, true);

					n_decode += 1;
				}
			}

			// all streams are finished
			if (batch.n_tokens == 0) {
				break;
			}

			cur_pos += 1;

			// evaluate the current batch with the transformer model
			if (llama_decode(ctx, batch)) {
				env->ThrowNew(IllegalStateException, "Failed to decode");
			}
		}

		llama_batch_free(batch);
		PERF_END("generate");
	}

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
	jjml_populate_default_samplers(model, smpl);

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
