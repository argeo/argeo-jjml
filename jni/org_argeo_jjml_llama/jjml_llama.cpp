#include <stddef.h>

#include "jjml_llama.h"

/*
 * SHARED C++ UTILITIES
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
