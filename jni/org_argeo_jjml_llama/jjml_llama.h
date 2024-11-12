#include <vector>

#include <llama.h>

#ifndef _jjml_llama_h
#define _jjml_llama_h

/*
 * SHARED UTILITIES
 */

/**
 * @brief Adds token to a batch.
 *
 * @param batch
 * @param id
 * @param pos
 * @param seq_ids
 * @param logits
 */
void jjml_llama_batch_add(struct llama_batch &batch, llama_token id,
		llama_pos pos, const std::vector<llama_seq_id> &seq_ids, bool logits);

/**
 * @brief Clears a batch.
 *
 * @param batch the batch to clear
 */
void jjml_llama_batch_clear(struct llama_batch &batch);

#endif
