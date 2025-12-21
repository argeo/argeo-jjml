/*
 * jjml_mtmd.h
 *
 *  Created on: 8 Sept 2025
 *      Author: mbaudier
 */

#ifndef JJML_MTMD_H_
#define JJML_MTMD_H_

#include <llama.h>
#include <mtmd.h>

/*
 * TODO factorize with llm?
 */
void jjml_mtmd_batch_add(struct llama_batch &batch, llama_token id,
		llama_pos pos, const std::vector<llama_seq_id> &seq_ids, bool logits);

void jjml_mtmd_batch_clear(struct llama_batch &batch);

int32_t jjml_mtmd_eval_chunks(mtmd_context *ctx, struct llama_context *lctx,
		const mtmd_input_chunks *chunks, llama_pos n_past, llama_seq_id seq_id,
		int32_t n_batch, bool logits_last, llama_pos *new_n_past);

#endif /* JJML_MTMD_H_ */
