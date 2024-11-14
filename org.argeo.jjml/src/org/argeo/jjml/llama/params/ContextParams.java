package org.argeo.jjml.llama.params;

import static java.lang.Boolean.parseBoolean;
import static java.lang.Integer.parseInt;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntSupplier;

//used by Javadoc only
import org.argeo.jjml.llama.LlamaCppContext;

/**
 * Parameters to configure a new context. New instance should be created by
 * using the {@link #with(Map)} methods on
 * {@link LlamaCppContext#defaultContextParams()}, with the default values
 * populated by the shared library.
 * 
 * Note: it provides record-style getters, in order to ease transition to Java
 * records in the future.
 */
public class ContextParams { //
	private final int n_ctx; // text context, 0 = from model
	private final int n_batch; // logical maximum batch size that can be submitted to llama_decode
	private final int n_ubatch; // physical maximum batch size
	private final int n_seq_max; // max number of sequences (i.e. distinct states for recurrent models)
	private final int n_threads; // number of threads to use for generation
	private final int n_threads_batch; // number of threads to use for batch processing

	private final int rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
	private final int pooling_type; // whether to pool (sum) embedding results by sequence id
	private final int attention_type; // attention type to use for embeddings

	// ref: https://github.com/ggerganov/llama.cpp/pull/2054
	private final float rope_freq_base; // RoPE base frequency, 0 = from model
	private final float rope_freq_scale; // RoPE frequency scaling factor, 0 = from model
	private final float yarn_ext_factor; // YaRN extrapolation mix factor, negative = from model
	private final float yarn_attn_factor; // YaRN magnitude scaling factor
	private final float yarn_beta_fast; // YaRN low correction dim
	private final float yarn_beta_slow; // YaRN high correction dim
	private final int yarn_orig_ctx; // YaRN original context size
	private final float defrag_thold; // defragment the KV cache if holes/size > thold, < 0 disabled (default)

	private final int type_k; // data type for K cache [EXPERIMENTAL]
	private final int type_v; // data type for V cache [EXPERIMENTAL]

	private final boolean embeddings; // if true, extract embeddings (together with logits)
	private final boolean offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
	private final boolean flash_attn; // whether to use flash attention [EXPERIMENTAL]
	private final boolean no_perf; // whether to measure performance timings

	/**
	 * Record-like full constructor. Will be called from the native side to provide
	 * the defaults.
	 */
	ContextParams(//
			int n_ctx, //
			int n_batch, //
			int n_ubatch, //
			int n_seq_max, //
			int n_threads, //
			int n_threads_batch, //
			int rope_scaling_type, //
			int pooling_type, //
			int attention_type, //
			float rope_freq_base, //
			float rope_freq_scale, //
			float yarn_ext_factor, //
			float yarn_attn_factor, //
			float yarn_beta_fast, //
			float yarn_beta_slow, //
			int yarn_orig_ctx, //
			float defrag_thold, //
			int type_k, //
			int type_v, //
			boolean embeddings, //
			boolean offload_kqv, //
			boolean flash_attn, //
			boolean no_perf //
	) {
		this.n_ctx = n_ctx;
		this.n_batch = n_batch;
		this.n_ubatch = n_ubatch;
		this.n_seq_max = n_seq_max;
		this.n_threads = n_threads;
		this.n_threads_batch = n_threads_batch;
		this.rope_scaling_type = rope_scaling_type;
		this.pooling_type = pooling_type;
		this.attention_type = attention_type;
		this.rope_freq_base = rope_freq_base;
		this.rope_freq_scale = rope_freq_scale;
		this.yarn_ext_factor = yarn_ext_factor;
		this.yarn_attn_factor = yarn_attn_factor;
		this.yarn_beta_fast = yarn_beta_fast;
		this.yarn_beta_slow = yarn_beta_slow;
		this.yarn_orig_ctx = yarn_orig_ctx;
		this.defrag_thold = defrag_thold;
		this.type_k = type_k;
		this.type_v = type_v;
		this.embeddings = embeddings;
		this.offload_kqv = offload_kqv;
		this.flash_attn = flash_attn;
		this.no_perf = no_perf;
	}

	public ContextParams with(ContextParam key, Object value) {
		Objects.requireNonNull(key);
		Objects.requireNonNull(value);
		String str;
		if (value instanceof IntSupplier) {
			// Needed by parameters defined as enums
			str = Integer.toString(((IntSupplier) value).getAsInt());
		} else {
			str = value.toString();
		}
		return with(Collections.singletonMap(key, str));
	}

	public ContextParams with(Map<ContextParam, String> p) {
		return new ContextParams( //
				parseInt(p.getOrDefault(ContextParam.n_ctx, Integer.toString(this.n_ctx))), //
				parseInt(p.getOrDefault(ContextParam.n_batch, Integer.toString(this.n_batch))), //
				parseInt(p.getOrDefault(ContextParam.n_ubatch, Integer.toString(this.n_ubatch))), //
				parseInt(p.getOrDefault(ContextParam.n_seq_max, Integer.toString(this.n_seq_max))), //
				parseInt(p.getOrDefault(ContextParam.n_threads, Integer.toString(this.n_threads))), //
				parseInt(p.getOrDefault(ContextParam.n_threads_batch, Integer.toString(this.n_threads_batch))), //
				this.rope_scaling_type, //
				parseInt(p.getOrDefault(ContextParam.pooling_type, Integer.toString(this.pooling_type))), //
				this.attention_type, //
				this.rope_freq_base, //
				this.rope_freq_scale, //
				this.yarn_ext_factor, //
				this.yarn_attn_factor, //
				this.yarn_beta_fast, //
				this.yarn_beta_slow, //
				this.yarn_orig_ctx, //
				this.defrag_thold, //
				this.type_k, //
				this.type_v, //
				parseBoolean(p.getOrDefault(ContextParam.embeddings, Boolean.toString(this.embeddings))), //
				this.offload_kqv, //
				this.flash_attn, //
				this.no_perf //
		);
	}

	public int n_ctx() {
		return n_ctx;
	}

	public int n_batch() {
		return n_batch;
	}

	public int n_ubatch() {
		return n_ubatch;
	}

	public int n_seq_max() {
		return n_seq_max;
	}

	public int n_threads() {
		return n_threads;
	}

	public int n_threads_batch() {
		return n_threads_batch;
	}

	public int rope_scaling_type() {
		return rope_scaling_type;
	}

	public int pooling_type() {
		return pooling_type;
	}

	public int attention_type() {
		return attention_type;
	}

	public float rope_freq_base() {
		return rope_freq_base;
	}

	public float rope_freq_scale() {
		return rope_freq_scale;
	}

	public float yarn_ext_factor() {
		return yarn_ext_factor;
	}

	public float yarn_attn_factor() {
		return yarn_attn_factor;
	}

	public float yarn_beta_fast() {
		return yarn_beta_fast;
	}

	public float yarn_beta_slow() {
		return yarn_beta_slow;
	}

	public int yarn_orig_ctx() {
		return yarn_orig_ctx;
	}

	public float defrag_thold() {
		return defrag_thold;
	}

	public int type_k() {
		return type_k;
	}

	public int type_v() {
		return type_v;
	}

	public boolean embeddings() {
		return embeddings;
	}

	public boolean offload_kqv() {
		return offload_kqv;
	}

	public boolean flash_attn() {
		return flash_attn;
	}

	public boolean no_perf() {
		return no_perf;
	}

//	/** Ensure that components and enum are perfectly in line. */
//	private static boolean assertParamNames() {
//		RecordComponent[] components = ContextParams.class.getRecordComponents();
//		ContextParamName[] names = ContextParamName.values();
//		assert components.length == names.length;
//		for (int i = 0; i < components.length; i++) {
//			assert components[i].getName().equals(names[i].name());
//		}
//		return true;
//	}
}
