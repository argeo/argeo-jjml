#include <stddef.h>
#include <algorithm>
#include <cassert>
#include <exception>
#include <stdexcept>
#include <vector>

#include <ggml.h>
#include <llama.h>

#include <argeo/jni/argeo_jni.h>

#include "jjml_llama.h"
#include "org_argeo_jjml_llama_LlamaCppBatchProcessor.h" // IWYU pragma: keep

/*
 * WRITE
 */
static jint jjml_llama_batch_processor_write(llama_context *ctx,
		llama_sampler *smpl, llama_pos cur_pos, void **inputs,
		const int inputs_count, JNIEnv *env, jintArray offsets,
		jintArray lengths, jintArray sequenceIds, jintArray outputIds,
		jboolean lastLogits) {
	assert(sequenceIds != nullptr);
	const int n_parallel = env->GetArrayLength(sequenceIds);

	const llama_model *model = llama_get_model(ctx);

	auto *sequence_ids =
			reinterpret_cast<llama_seq_id*>(env->GetIntArrayElements(
					sequenceIds, nullptr));
	auto *output_ids = reinterpret_cast<int32_t*>(env->GetIntArrayElements(
			outputIds, nullptr));

	assert(inputs_count > 0);
	assert(lengths != nullptr);
	assert(env->GetArrayLength(lengths) == inputs_count);
	jint *seq_tokens_size = env->GetIntArrayElements(lengths, nullptr);
	assert(offsets != nullptr);
	assert(env->GetArrayLength(offsets) == inputs_count);
	jint *seq_offsets = env->GetIntArrayElements(offsets, nullptr);

	llama_token *seq_tokens[inputs_count];
	for (int i = 0; i < inputs_count; i++) {
		void *input = inputs[i];
		if (input != nullptr) {
			seq_tokens[i] = static_cast<llama_token*>(input) + seq_offsets[i];
		} else {
			seq_tokens[i] = nullptr;
		}
	}

	int total_tokens = 0;
	int max_decodes = 0;
	for (int i = 0; i < inputs_count; i++) {
		total_tokens = total_tokens + seq_tokens_size[i];
		if (seq_tokens_size[i] > max_decodes)
			max_decodes = seq_tokens_size[i];
	}

	PERF_BEGIN();
	if (inputs_count == 1) { // common prompt to all sequences
		const int seq_idx = 0;
		int input_tokens_size = seq_tokens_size[seq_idx];

		llama_batch batch = llama_batch_init(
				std::max((size_t) input_tokens_size, (size_t) n_parallel), 0,
				n_parallel);

		for (size_t i = 0; i < input_tokens_size; i++) {
			// FIXME deal with null input
			batch.token[batch.n_tokens] = seq_tokens[seq_idx][i];
			batch.pos[batch.n_tokens] = cur_pos + i;
			batch.n_seq_id[batch.n_tokens] = n_parallel;
			for (size_t j = 0; j < n_parallel; j++) {
				batch.seq_id[batch.n_tokens][j] = sequence_ids[j];
			}
			batch.logits[batch.n_tokens] = false;

			batch.n_tokens++;

		}
		batch.n_tokens = input_tokens_size;
		GGML_ASSERT(batch.n_tokens == (int ) input_tokens_size);

		// FIXME temporary backward compatibility
		std::vector<llama_seq_id> seq_ids(n_parallel, 0);
		for (int i = 0; i < n_parallel; i++) {
			seq_ids[i] = sequence_ids[i];
		}

		// deal with encoder
		// TODO does it make sense to do that when it is not the first batch?
		if (llama_model_has_encoder(model)) {
			if (llama_encode(ctx, batch))
				throw std::runtime_error("Encode failed");

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

		if (llama_decode(ctx, batch) != 0)
			throw std::runtime_error("Decode failed");

		// sampler accept
		for (int i = 0; i < batch.n_tokens; i++) {
			llama_token token = batch.token[i];
			llama_sampler_accept(smpl, token);
		}

		cur_pos = cur_pos + batch.n_tokens;
		llama_batch_free(batch);

	} else {
		assert(inputs_count == n_parallel);

		llama_batch batch = llama_batch_init(total_tokens, 0, n_parallel);
		try {
			for (size_t j = 0; j < n_parallel; j++) {
				for (size_t i = 0; i < seq_tokens_size[j]; i++) {
					// FIXME deal with null input
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
			if (llama_decode(ctx, batch) != 0)
				throw std::runtime_error("Decode failed");
		} catch (...) {
			llama_batch_free(batch);
			throw std::current_exception();
		}
		llama_batch_free(batch);
	}

	PERF_END(__func__);

	// clean up
	env->ReleaseIntArrayElements(offsets, reinterpret_cast<jint*>(seq_offsets),
			0);
	env->ReleaseIntArrayElements(lengths,
			reinterpret_cast<jint*>(seq_tokens_size), 0);
	env->ReleaseIntArrayElements(sequenceIds,
			reinterpret_cast<jint*>(sequence_ids), 0);
	env->ReleaseIntArrayElements(outputIds, reinterpret_cast<jint*>(output_ids),
			0);

	return cur_pos;
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doWrite(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerChainPointer,
		jint contextPosition, jobjectArray inputBuffers, jintArray offsets,
		jintArray lengths, jintArray sequenceIds, jintArray outputIds,
		jboolean lastLogits) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(samplerChainPointer);
	llama_pos cur_pos = static_cast<llama_pos>(contextPosition);

	int inputs_count = env->GetArrayLength(inputBuffers);
	void *inputs[inputs_count];
	for (int i = 0; i < inputs_count; i++) {
		jobject inputBuf = env->GetObjectArrayElement(inputBuffers, i);
		if (inputBuf != nullptr) {
			inputs[i] = env->GetDirectBufferAddress(inputBuf);
		} else {
			inputs[i] = nullptr;
		}
	}

	jint newPosition;
	try {
		newPosition = jjml_llama_batch_processor_write(ctx, smpl, cur_pos,
				inputs, inputs_count, env, offsets, lengths, sequenceIds,
				outputIds, lastLogits);
	} catch (std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
	}
	return newPosition;
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppBatchProcessor_doWriteArrays(
		JNIEnv *env, jclass, jlong contextPointer, jlong samplerChainPointer,
		jint contextPosition, jobjectArray inputArrays, jintArray offsets,
		jintArray lengths, jintArray sequenceIds, jintArray outputIds,
		jboolean lastLogits) {
	auto *ctx = argeo::jni::as_pointer<llama_context*>(contextPointer);
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(samplerChainPointer);
	llama_pos cur_pos = static_cast<llama_pos>(contextPosition);

	int inputs_count = env->GetArrayLength(inputArrays);
	void *inputs[inputs_count];
	for (int i = 0; i < inputs_count; i++) {
		jarray arr = (jarray) env->GetObjectArrayElement(inputArrays, i);
		if (arr != nullptr) {
			inputs[i] = env->GetPrimitiveArrayCritical(arr, nullptr);
		} else {
			inputs[i] = nullptr;
		}
	}

	jint newPosition;
	try {
		newPosition = jjml_llama_batch_processor_write(ctx, smpl, cur_pos,
				inputs, inputs_count, env, offsets, lengths, sequenceIds,
				outputIds, lastLogits);
	} catch (std::exception &ex) {
		argeo::jni::throw_to_java(env, ex);
	}

	// clean up
	for (int i = 0; i < inputs_count; i++) {
		jarray arr = (jarray) env->GetObjectArrayElement(inputArrays, i);
		if (arr != nullptr) {
			env->ReleasePrimitiveArrayCritical(arr, inputs[i], 0);
		}
	}

	return newPosition;
}
