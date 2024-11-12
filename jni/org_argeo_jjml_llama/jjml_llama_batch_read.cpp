#include <math.h>
#include <cassert>
#include <stdexcept>
#include <vector>

#include <ggml.h>
#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "jjml_llama.h"
#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppBatchProcessor.h" // IWYU pragma: keep

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

static jint jjml_llama_batch_processor_read(llama_context *ctx,
		llama_sampler *smpl, llama_sampler *grmr, llama_pos cur_pos,
		void **outputs, const int outputs_count, JNIEnv *env, jintArray offsets,
		jintArray lengths, jintArray sequenceIds, jintArray outputIds,
		jobject completionHandler) {

	const llama_model *model = llama_get_model(ctx);

	const uint32_t NO_OUTPUT_ID = llama_n_batch(ctx);

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

	auto *output_ids = reinterpret_cast<int32_t*>(env->GetIntArrayElements(
			outputIds, nullptr));

//	const int outBuffersCount = env->GetArrayLength(outputBuffers);
	assert(outputs_count == n_parallel && "As many buffers as sequences");

	assert(outputs_count > 0);
	assert(lengths != nullptr);
	assert(env->GetArrayLength(lengths) == outputs_count);
	jint *seq_tokens_size = env->GetIntArrayElements(lengths, nullptr);
	assert(offsets != nullptr);
	assert(env->GetArrayLength(offsets) == outputs_count);
	jint *seq_offsets = env->GetIntArrayElements(offsets, nullptr);

	llama_token *seq_tokens[outputs_count];
	for (int i = 0; i < outputs_count; i++) {
		void *output = outputs[i];
		if (output != nullptr) {
			seq_tokens[i] = static_cast<llama_token*>(output) + seq_offsets[i];
		} else {
			seq_tokens[i] = nullptr;
		}
	}

	int max_decodes = 0;
	for (int i = 0; i < n_parallel; i++) {
		if (seq_tokens_size[i] > max_decodes)
			max_decodes = seq_tokens_size[i];
	}

	PERF_BEGIN();

	int next_idx = 0;

	llama_batch batch = llama_batch_init(n_parallel, 0, n_parallel);

	// FIXME deal more precisely with the upper limit
	int n_predict = cur_pos + max_decodes;
	bool all_eog = true;
	while (cur_pos <= n_predict) {
		// prepare the next batch
		jjml_llama_batch_clear(batch);

		// sample the next token for each parallel sequence / stream
		for (int32_t i = 0; i < n_parallel; ++i) {
			if (output_ids[i] == NO_OUTPUT_ID) {
				// the stream has already finished
				continue;
			}

			//PERF_BEGIN();
			llama_token new_token_id;
			if (grmr == nullptr) {
				new_token_id = llama_sampler_sample(smpl, ctx, output_ids[i]);

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

				jclass Integer = argeo::jni::find_jclass(env,
						"java/lang/Integer");
				jobject completionHandlerResult = env->CallStaticObjectMethod(
						Integer, Integer$valueOf, next_idx);
				jobject completionHandlerAttachment =
						env->CallStaticObjectMethod(Integer, Integer$valueOf,
								i);
				// call completion handler
				env->CallVoidMethod(completionHandler,
						CompletionHandler$completed, completionHandlerResult,
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
			jjml_llama_batch_add(batch, new_token_id, cur_pos,
					{ sequence_ids[i] }, true);
		}
		next_idx++;

		// all streams are finished
		if (batch.n_tokens == 0) {
			break;
		}

		cur_pos += 1;

		// evaluate the current batch with the transformer model
		if (llama_decode(ctx, batch))
			throw std::runtime_error("Decode failed");
	}

	// clean up
// !! dereference what Java owns before batch is freed
//	for (size_t i = 0; i < n_parallel; i++) {
//		batch.seq_id[i] = nullptr;
//	}
	llama_batch_free(batch);
	PERF_END(__func__);

	// clean up
	env->ReleaseIntArrayElements(offsets, reinterpret_cast<jint*>(seq_offsets),
			0);
	env->ReleaseIntArrayElements(lengths,
			reinterpret_cast<jint*>(seq_tokens_size), 0);
	env->ReleaseIntArrayElements(outputIds, reinterpret_cast<jint*>(output_ids),
			0);

	// TODO assert consistency of context position with regard to the output buffers sizes
	return cur_pos;
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doRead(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerPtr,
		jlong grammarSamplerPtr, jint contextPosition,
		jobjectArray outputBuffers, jintArray offsets, jintArray lengths,
		jintArray sequenceIds, jintArray outputIds, jobject completionHandler) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);

	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(samplerPtr);
	auto *grmr =
			grammarSamplerPtr != 0 ?
					argeo::jni::as_pointer<llama_sampler*>(grammarSamplerPtr) :
					nullptr;
	llama_pos cur_pos = static_cast<llama_pos>(contextPosition);

	int outputs_count = env->GetArrayLength(outputBuffers);
	void *outputs[outputs_count];
	for (int i = 0; i < outputs_count; i++) {
		jobject buf = env->GetObjectArrayElement(outputBuffers, i);
		if (buf != nullptr) {
			outputs[i] = env->GetDirectBufferAddress(buf);
		} else {
			outputs[i] = nullptr;
		}
	}

	jint newPosition;
	try {
		newPosition = jjml_llama_batch_processor_read(ctx, smpl, grmr, cur_pos,
				outputs, outputs_count, env, offsets, lengths, sequenceIds,
				outputIds, completionHandler);
	} catch (std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
	}
	return newPosition;
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doReadToArrays(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerPtr,
		jlong grammarSamplerPtr, jint contextPosition,
		jobjectArray outputArrays, jintArray offsets, jintArray lengths,
		jintArray sequenceIds, jintArray outputIds, jobject completionHandler) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);

	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(samplerPtr);
	auto *grmr =
			grammarSamplerPtr != 0 ?
					argeo::jni::as_pointer<llama_sampler*>(grammarSamplerPtr) :
					nullptr;
	llama_pos cur_pos = static_cast<llama_pos>(contextPosition);

	int outputs_count = env->GetArrayLength(outputArrays);
	void *outputs[outputs_count];
	for (int i = 0; i < outputs_count; i++) {
		jintArray arr = (jintArray) env->GetObjectArrayElement(outputArrays, i);
		if (arr != nullptr) {
			outputs[i] = env->GetPrimitiveArrayCritical(arr, nullptr);
//			outputs[i] = env->GetIntArrayElements(arr, nullptr);
		} else {
			outputs[i] = nullptr;
		}
	}

	jint newPosition;
	try {
		newPosition = jjml_llama_batch_processor_read(ctx, smpl, grmr, cur_pos,
				outputs, outputs_count, env, offsets, lengths, sequenceIds,
				outputIds, completionHandler);
	} catch (std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
	}

	// clean up
	for (int i = 0; i < outputs_count; i++) {
		jintArray arr = (jintArray) env->GetObjectArrayElement(outputArrays, i);
		if (arr != nullptr) {
			env->ReleasePrimitiveArrayCritical(arr, outputs[i], 0);
//			env->ReleaseIntArrayElements(arr, (int*) outputs[i], 0);
		}
	}

	return newPosition;
}

