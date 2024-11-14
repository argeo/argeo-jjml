package org.argeo.jjml.llama.params;

/** Supported parameters. */
public enum ContextParam {
	n_ctx, //
	n_batch, //
	n_ubatch, //
	n_seq_max, //
	n_threads, //
	n_threads_batch, //
//		rope_scaling_type, //
	pooling_type, //
//		attention_type, //
//		rope_freq_base, //
//		rope_freq_scale, //
//		yarn_ext_factor, //
//		yarn_attn_factor, //
//		yarn_beta_fast, //
//		yarn_beta_slow, //
//		yarn_orig_ctx, //
//		defrag_thold, //
//		type_k, //
//		type_v, //
	embeddings, //
//		offload_kqv, //
//		flash_attn, //
//		no_perf, //
	;

	/** System property to set the default context size. */
	public final static String SYSTEM_PROPERTY_CONTEXT_PARAM_PREFIX = "jjml.llama.context.";

}
