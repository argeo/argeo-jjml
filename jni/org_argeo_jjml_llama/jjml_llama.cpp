#include "jjml_llama.h"

#include <argeo/argeo_jni.h>
#include <ggml.h>
#include <jni.h>
#include <jni_md.h>
#include <stddef.h>
#include <cassert>

#include "org_argeo_jjml_llama_LlamaCppBackend.h" // IWYU pragma: keep

/*
 * BACKEND
 */
JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_doInit(
		JNIEnv *env, jclass) {
	llama_backend_init();
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_doNumaInit(
		JNIEnv*, jclass, jint numaStrategy) {
	switch (numaStrategy) {
	case GGML_NUMA_STRATEGY_DISABLED:
		llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
		break;
	case GGML_NUMA_STRATEGY_DISTRIBUTE:
		llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
		break;
	case GGML_NUMA_STRATEGY_ISOLATE:
		llama_numa_init(GGML_NUMA_STRATEGY_ISOLATE);
		break;
	case GGML_NUMA_STRATEGY_NUMACTL:
		llama_numa_init(GGML_NUMA_STRATEGY_NUMACTL);
		break;
	case GGML_NUMA_STRATEGY_MIRROR:
		llama_numa_init(GGML_NUMA_STRATEGY_MIRROR);
		break;
	default:
		assert(!"Invalid NUMA strategy enum value");
		break;
	}
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppBackend_doDestroy(
		JNIEnv*, jclass) {
	llama_backend_free();
}

/*
 * BATCH UTILITIES
 */
void jjml_llama_batch_add(struct llama_batch &batch, llama_token id,
		llama_pos pos, const std::vector<llama_seq_id> &seq_ids, bool logits) {
	batch.token[batch.n_tokens] = id;
	batch.pos[batch.n_tokens] = pos;
	batch.n_seq_id[batch.n_tokens] = seq_ids.size();
	for (size_t i = 0; i < seq_ids.size(); ++i) {
		batch.seq_id[batch.n_tokens][i] = seq_ids[i];
	}
	batch.logits[batch.n_tokens] = logits;

	batch.n_tokens++;
}

void jjml_llama_batch_clear(struct llama_batch &batch) {
	batch.n_tokens = 0;
}
