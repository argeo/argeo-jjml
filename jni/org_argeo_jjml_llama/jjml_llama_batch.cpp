#include <argeo/argeo_jni.h>
#include <ggml.h>
#include <jni.h>
#include <jni_md.h>
#include <llama.h>
#include <stddef.h>
#include <algorithm>
#include <vector>
#include <math.h>
#include <cassert>

#include "jjml_llama.h"
#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppBatchProcessor.h" // IWYU pragma: keep

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doWriteBatch(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerChainPointer,
		jint contextPosition, jobjectArray inputBuffers, jintArray sequenceIds,
		jintArray outputIds, jboolean lastLogits) {
	auto *ctx = argeo::jni::getPointer<llama_context*>(contextPointer);
	const llama_model *model = llama_get_model(ctx);

	auto *smpl = argeo::jni::getPointer<llama_sampler*>(samplerChainPointer);

	int cur_pos = contextPosition;

	const int n_parallel = env->GetArrayLength(sequenceIds);
	auto *sequence_ids = reinterpret_cast<llama_seq_id*>(env->GetIntArrayElements(
			sequenceIds, nullptr));
	auto *output_ids = reinterpret_cast<int32_t*>(env->GetIntArrayElements(outputIds,
			nullptr));

	int inputBuffersCount = env->GetArrayLength(inputBuffers);
	assert(inputBuffersCount > 0 && "Input buffers count");

	if (inputBuffersCount == 1) { // common prompt to all sequences

		jobject inputBuf = env->GetObjectArrayElement(inputBuffers, 0);

		PERF_BEGIN();

//		int input_tokens_size = env->GetDirectBufferCapacity(inputBuf);
		int input_tokens_size = env->CallIntMethod(inputBuf, IntBuffer$limit);
		llama_token *tokens =
				static_cast<llama_token*>(env->GetDirectBufferAddress(inputBuf));

		llama_batch batch = llama_batch_init(
				std::max((size_t) input_tokens_size, (size_t) n_parallel), 0,
				n_parallel);

		// Prepare the initial batch
		// !! we use the Java direct buffer as source for the input tokens
//		free(batch.token);
//		free(batch.seq_id);
//		batch.token = input_tokens;
		for (size_t i = 0; i < input_tokens_size; i++) {
			batch.token[batch.n_tokens] = tokens[i];
			batch.pos[batch.n_tokens] = cur_pos + i;
			batch.n_seq_id[batch.n_tokens] = n_parallel;
			for (size_t j = 0; j < n_parallel; j++) {
				batch.seq_id[batch.n_tokens][j] = sequence_ids[j];
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

		// deal with encoder
		// TODO does it make sense to do that when it is not the first batch?
		if (llama_model_has_encoder(model)) {
			if (llama_encode(ctx, batch)) {
				env->ThrowNew(IllegalStateException, "Failed to encode");
				return -1;
			}

			llama_token decoder_start_token_id =
					llama_model_decoder_start_token(model);
			if (decoder_start_token_id == -1) {
				decoder_start_token_id = llama_token_bos(model);
			}

			jjml_llama_batch_clear(batch);
			jjml_llama_batch_add(batch, decoder_start_token_id, cur_pos,
					seq_ids, false);
		}

		// llama_decode will output logits only for the last token of the prompt
		if (lastLogits) {
			batch.logits[batch.n_tokens - 1] = true;
			for (int i = 0; i < n_parallel; i++) {
				output_ids[i] = batch.n_tokens - 1;
			}
		}

		if (llama_decode(ctx, batch) != 0) {
			env->ThrowNew(IllegalStateException, "Decode failed");
		}

		// sampler accept
		for (int i = 0; i < batch.n_tokens; i++) {
			llama_token token = batch.token[i];
			llama_sampler_accept(smpl, token);
		}

		cur_pos = cur_pos + batch.n_tokens;

		// !! dereference what Java owns before batch is freed
//		batch.token = nullptr;
//		for (size_t i = 0; i < input_tokens_size; i++) {
//			batch.seq_id[i] = nullptr;
//		}
		llama_batch_free(batch);

		PERF_END(__func__);
	} else {
		assert(
				inputBuffersCount == n_parallel
						&& "As many buffers as sequences");

		llama_token *seq_tokens[n_parallel];
		int seq_tokens_size[n_parallel];
		int total_tokens = 0;
		int max_decodes = 0;

		for (int i = 0; i < n_parallel; i++) {
			jobject inputBuf = env->GetObjectArrayElement(inputBuffers, i);
			if (inputBuf != nullptr) {
				void *input = env->GetDirectBufferAddress(inputBuf);
				int input_tokens_size = env->CallIntMethod(inputBuf,
						IntBuffer$limit);
				assert(input_tokens_size > 0 && "Input buffer capacity");

				llama_token *input_tokens = static_cast<llama_token*>(input);

				seq_tokens[i] = input_tokens;
				seq_tokens_size[i] = input_tokens_size;
				total_tokens = total_tokens + input_tokens_size;
				if (input_tokens_size > max_decodes)
					max_decodes = input_tokens_size;
			}
		}

		llama_batch batch = llama_batch_init(total_tokens, 0, n_parallel);

		for (size_t j = 0; j < n_parallel; j++) {
			for (size_t i = 0; i < seq_tokens_size[j]; i++) {
				batch.token[batch.n_tokens] = seq_tokens[j][i];
				batch.pos[batch.n_tokens] = cur_pos;
				batch.n_seq_id[batch.n_tokens] = 1;
				batch.seq_id[batch.n_tokens][0] = sequence_ids[j];
				batch.logits[batch.n_tokens] = false;
				batch.n_tokens++;
				cur_pos++;
			}

			if (lastLogits) {
				batch.logits[batch.n_tokens - 1] = true;
				output_ids[j] = batch.n_tokens - 1;
			}
		}

		// TODO deal with encoder models?

		if (llama_decode(ctx, batch) != 0) {
			env->ThrowNew(IllegalStateException, "Decode failed");
		}

		llama_batch_free(batch);
	}
	// clean up
	env->ReleaseIntArrayElements(sequenceIds, reinterpret_cast<jint*>(sequence_ids), 0);
	env->ReleaseIntArrayElements(outputIds, reinterpret_cast<jint*>(output_ids), 0);

	return cur_pos;
}

static std::vector<llama_token_data> jjml_get_logits(llama_context *ctx,
		int idx) {
	const auto *logits = llama_get_logits_ith(ctx, idx);

	const int n_vocab = llama_n_vocab(llama_get_model(ctx));

	std::vector<llama_token_data> cur;
	cur.resize(n_vocab);

	for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
		cur[token_id] = llama_token_data { token_id, logits[token_id], 0.0f };
	}

	return cur;
}

static llama_token jjml_check_grammar(llama_context *ctx, int idx,
		llama_sampler *chain, llama_sampler *grmr, llama_token id) {
	// check if it the sampled token fits the grammar
	{
		llama_token_data single_token_data = { id, 1.0f, 0.0f };
		llama_token_data_array single_token_data_array = { &single_token_data,
				1, -1, false };

		llama_sampler_apply(grmr, &single_token_data_array);

		const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
		if (is_valid) {
			return id;
		}
	}

	// resampling:
	// if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
	std::vector<llama_token_data> cur = jjml_get_logits(ctx, idx);
	llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false, };

	llama_sampler_apply(grmr, &cur_p);
	llama_sampler_apply(chain, &cur_p);

	GGML_ASSERT(
			cur_p.selected != -1
					&& "no selected token during re-sampling - check your sampling configuration");

	llama_token res = cur_p.data[cur_p.selected].id;
	return res;
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doReadBatch(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerPtr,
		jlong grammarSamplerPtr, jint contextPosition,
		jobjectArray outputBuffers, jintArray sequenceIds, jintArray outputIds,
		jobject completionHandler) {
	auto *ctx = argeo::jni::getPointer<llama_context*>(contextPointer);
	const llama_model *model = llama_get_model(ctx);

	auto *smpl = argeo::jni::getPointer<llama_sampler*>(samplerPtr);
	auto *grmr =
			grammarSamplerPtr != 0 ?
					argeo::jni::getPointer<llama_sampler*>(grammarSamplerPtr) :
					nullptr;

	const uint32_t NO_OUTPUT_ID = llama_n_batch(ctx);

	llama_pos cur_pos = static_cast<llama_pos>(contextPosition);

	const int n_parallel = env->GetArrayLength(sequenceIds);
	assert(n_parallel > 0 && "Sequence count");
//	auto *sequence_ids = static_cast<llama_seq_id*>(env->GetIntArrayElements(
//			sequenceIds, nullptr));
	jint *arr = env->GetIntArrayElements(sequenceIds, nullptr);
	llama_seq_id sequence_ids[n_parallel];
	for (int i = 0; i < n_parallel; i++) {
		sequence_ids[i] = static_cast<llama_seq_id>(arr[i]);
	}
	env->ReleaseIntArrayElements(sequenceIds, arr, 0);

	auto *output_ids = reinterpret_cast<int32_t*>(env->GetIntArrayElements(outputIds,
			nullptr));

	const int outBuffersCount = env->GetArrayLength(outputBuffers);
	assert(outBuffersCount == n_parallel && "As many buffers as sequences");

	llama_token *seq_tokens[n_parallel];
	int seq_tokens_size[n_parallel];
	int max_decodes = 0;

	for (int i = 0; i < n_parallel; i++) {
		jobject outputBuf = env->GetObjectArrayElement(outputBuffers, i);
		if (outputBuf != nullptr) {
			void *output = env->GetDirectBufferAddress(outputBuf);
			int out_tokens_size = env->CallIntMethod(outputBuf,
					IntBuffer$limit);
			assert(out_tokens_size > 0 && "Output buffer capacity");

			llama_token *output_tokens = static_cast<llama_token*>(output);

			seq_tokens[i] = output_tokens;
			seq_tokens_size[i] = out_tokens_size;
			if (out_tokens_size > max_decodes)
				max_decodes = out_tokens_size;
		}
	}

	PERF_BEGIN();

	int next_idx = 0;

	llama_batch batch = llama_batch_init(n_parallel, 0, n_parallel);

// FIXME deal more precisely with the upper limit
//assert(max_decodes % sizeof(llama_token) == 0);
	int n_predict = cur_pos + max_decodes;
	bool all_eog = true;
	while (cur_pos <= n_predict) {
		// prepare the next batch
		jjml_llama_batch_clear(batch);

		{
			// sample the next token for each parallel sequence / stream
			for (int32_t i = 0; i < n_parallel; ++i) {
				if (output_ids[i] == NO_OUTPUT_ID) {
					// the stream has already finished
					continue;
				}

				//PERF_BEGIN();
				llama_token new_token_id;
				if (grmr == nullptr) {
					new_token_id = llama_sampler_sample(smpl, ctx,
							output_ids[i]);

				} else {	// grammar handling require lower-level methods
					std::vector<llama_token_data> cur = jjml_get_logits(ctx,
							output_ids[i]);
					llama_token_data_array cur_p = { cur.data(), cur.size(), -1,
							false, };

					llama_sampler_apply(smpl, &cur_p);
					llama_token candidate = cur_p.data[cur_p.selected].id;
					new_token_id = jjml_check_grammar(ctx, output_ids[i], smpl,
							grmr, candidate);
//					new_token_id = candidate;

					llama_sampler_accept(grmr, new_token_id);
					llama_sampler_accept(smpl, new_token_id);
				}
				//PERF_END("sampling");

				bool is_eog = llama_token_is_eog(model, new_token_id);

				// is it an end of generation? -> mark the stream as finished
				if (is_eog //
				|| next_idx == seq_tokens_size[i] //
				|| cur_pos == n_predict //
						) {
					if (is_eog)
						output_ids[i] = NO_OUTPUT_ID;

					jobject outputBuf = env->GetObjectArrayElement(
							outputBuffers, i);

					// set output buffer in a proper state
					env->CallVoidMethod(outputBuf, IntBuffer$limitI, next_idx);
					env->CallVoidMethod(outputBuf, IntBuffer$positionI,
							next_idx);

					// TODO find out while call to static method is failing with centralized class
					jclass Integer = env->FindClass("java/lang/Integer");

					jobject completionHandlerResult =
							env->CallStaticObjectMethod(Integer,
									Integer$valueOf, next_idx);
					jobject completionHandlerAttachment =
							env->CallStaticObjectMethod(Integer,
									Integer$valueOf, i);
					// call completion handler
					env->CallVoidMethod(completionHandler,
							CompletionHandler$completed,
							completionHandlerResult,
							completionHandlerAttachment);

					if (!is_eog) // at least one could have continued
						all_eog = false;

					continue;
				}

//				std::cerr << cur_pos << "\t" << i << "\t" << new_token_id
//						<< std::endl;

				assert(next_idx < seq_tokens_size[i] && "No overflow");
				seq_tokens[i][next_idx] = new_token_id;

				output_ids[i] = batch.n_tokens;

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

	// clean up
// !! dereference what Java owns before batch is freed
//	for (size_t i = 0; i < n_parallel; i++) {
//		batch.seq_id[i] = nullptr;
//	}
	llama_batch_free(batch);
	PERF_END(__func__);

// clean up
	env->ReleaseIntArrayElements(outputIds, reinterpret_cast<jint*>(output_ids), 0);

	// TODO assert consistency of context position with regard to the output buffers sizes
	return cur_pos;
}
