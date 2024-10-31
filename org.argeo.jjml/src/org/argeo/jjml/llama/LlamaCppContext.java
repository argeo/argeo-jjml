package org.argeo.jjml.llama;

import static java.lang.Boolean.parseBoolean;
import static java.lang.Integer.parseInt;

import java.lang.reflect.RecordComponent;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntSupplier;
import java.util.function.LongSupplier;

/**
 * Access to a llama.cpp context
 * 
 * @see llama.h - llama_context
 */
public class LlamaCppContext implements LongSupplier, AutoCloseable {
	public final static Params DEFAULT_PARAMS;

	static {
		// TODO implement missing parameters
		// assert Params.assertParamNames();
		DEFAULT_PARAMS = Params.defaultContextParams();
	}

	private final long pointer;
	private final LlamaCppModel model;

	private final LlamaCppContext.Params initParams;

	// Effective parameters
	private final LlamaCppPoolingType poolingType;
	private final int contextSize;
	private final int batchSize;

	public LlamaCppContext(LlamaCppModel model) {
		this(model, DEFAULT_PARAMS);
	}

	public LlamaCppContext(LlamaCppModel model, LlamaCppContext.Params initParams) {
		Objects.requireNonNull(model);
		Objects.requireNonNull(initParams);
		this.pointer = doInit(model, initParams);
		this.model = model;
		this.initParams = initParams;
		int poolingTypeCode = doGetPoolingType();
		poolingType = LlamaCppPoolingType.byCode(poolingTypeCode);
		contextSize = doGetContextSize();
		batchSize = doGetBatchSize();
	}

	/*
	 * NATIVE
	 */
	private static native long doInit(LlamaCppModel model, LlamaCppContext.Params params);

	native void doDestroy();

	private native int doGetPoolingType();

	private native int doGetContextSize();

	private native int doGetBatchSize();

	/*
	 * LIFECYCLE
	 */
	@Override
	public void close() throws RuntimeException {
		doDestroy();
	}

	/*
	 * ACCESSORS
	 */
	@Override
	public long getAsLong() {
		return pointer;
	}

	public LlamaCppModel getModel() {
		return model;
	}

	public LlamaCppContext.Params getInitParams() {
		return initParams;
	}

	public LlamaCppPoolingType getPoolingType() {
		return poolingType;
	}

	public int getContextSize() {
		return contextSize;
	}

	public int getBatchSize() {
		return batchSize;
	}

	/*
	 * CLASSES
	 */
	/** Supported parameters. */
	public static enum ParamName {
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
	}

	public static record Params( //
			int n_ctx, // text context, 0 = from model
			int n_batch, // logical maximum batch size that can be submitted to llama_decode
			int n_ubatch, // physical maximum batch size
			int n_seq_max, // max number of sequences (i.e. distinct states for recurrent models)
			int n_threads, // number of threads to use for generation
			int n_threads_batch, // number of threads to use for batch processing

			int rope_scaling_type, // RoPE scaling type, from `enum llama_rope_scaling_type`
			int pooling_type, // whether to pool (sum) embedding results by sequence id
			int attention_type, // attention type to use for embeddings

			// ref: https://github.com/ggerganov/llama.cpp/pull/2054
			float rope_freq_base, // RoPE base frequency, 0 = from model
			float rope_freq_scale, // RoPE frequency scaling factor, 0 = from model
			float yarn_ext_factor, // YaRN extrapolation mix factor, negative = from model
			float yarn_attn_factor, // YaRN magnitude scaling factor
			float yarn_beta_fast, // YaRN low correction dim
			float yarn_beta_slow, // YaRN high correction dim
			int yarn_orig_ctx, // YaRN original context size
			float defrag_thold, // defragment the KV cache if holes/size > thold, < 0 disabled (default)

			int type_k, // data type for K cache [EXPERIMENTAL]
			int type_v, // data type for V cache [EXPERIMENTAL]

			boolean embeddings, // if true, extract embeddings (together with logits)
			boolean offload_kqv, // whether to offload the KQV ops (including the KV cache) to GPU
			boolean flash_attn, // whether to use flash attention [EXPERIMENTAL]
			boolean no_perf // whether to measure performance timings
	) {
		public Params with(ParamName key, Object value) {
			Objects.requireNonNull(key);
			Objects.requireNonNull(value);
			String str;
			if (value instanceof IntSupplier supplier) {
				str = Integer.toString(supplier.getAsInt());
			} else {
				str = value.toString();
			}
			return with(Collections.singletonMap(key, str));
		}

		public Params with(Map<ParamName, String> p) {
			return new Params( //
					parseInt(p.getOrDefault(ParamName.n_ctx, Integer.toString(this.n_ctx))), //
					parseInt(p.getOrDefault(ParamName.n_batch, Integer.toString(this.n_batch))), //
					parseInt(p.getOrDefault(ParamName.n_ubatch, Integer.toString(this.n_ubatch))), //
					parseInt(p.getOrDefault(ParamName.n_seq_max, Integer.toString(this.n_seq_max))), //
					parseInt(p.getOrDefault(ParamName.n_threads, Integer.toString(this.n_threads))), //
					parseInt(p.getOrDefault(ParamName.n_threads_batch, Integer.toString(this.n_threads_batch))), //
					this.rope_scaling_type, //
					parseInt(p.getOrDefault(ParamName.pooling_type, Integer.toString(this.pooling_type))), //
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
					parseBoolean(p.getOrDefault(ParamName.embeddings, Boolean.toString(this.embeddings))), //
					this.offload_kqv, //
					this.flash_attn, //
					this.no_perf //
			);
		}

		/**
		 * The default context parameters.
		 * 
		 * @see llama.h - llama_context_default_params()
		 */
		public static Params defaultContextParams() {
			LlamaCppNative.ensureLibrariesLoaded();
			return LlamaCppNative.newContextParams();
		}

		/** Ensure that components and enum are perfectly in line. */
		private static boolean assertParamNames() {
			RecordComponent[] components = Params.class.getRecordComponents();
			ParamName[] names = ParamName.values();
			assert components.length == names.length;
			for (int i = 0; i < components.length; i++) {
				assert components[i].getName().equals(names[i].name());
			}
			return true;
		}
	}

}
