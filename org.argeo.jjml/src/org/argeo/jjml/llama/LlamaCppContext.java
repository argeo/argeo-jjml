package org.argeo.jjml.llama;

import java.util.Objects;
import java.util.function.LongSupplier;

import org.argeo.jjml.llama.params.ContextParams;
import org.argeo.jjml.llama.params.PoolingType;

/**
 * Access to a llama.cpp context
 * 
 * @see llama.h - llama_context
 */
public class LlamaCppContext implements LongSupplier, AutoCloseable {
	private final static ContextParams DEFAULT_PARAMS;

	static {
		// TODO implement missing parameters
		// assert Params.assertParamNames();
		DEFAULT_PARAMS = LlamaCppNative.newContextParams();
	}

	private final long pointer;
	private final LlamaCppModel model;

	private final ContextParams initParams;

	// Effective parameters
	private final PoolingType poolingType;
	private final int contextSize;
	private final int batchSize;

	private LlamaCppBatchProcessor batchProcessor;

	public LlamaCppContext(LlamaCppModel model) {
		this(model, DEFAULT_PARAMS);
	}

	public LlamaCppContext(LlamaCppModel model, ContextParams initParams) {
		Objects.requireNonNull(model);
		Objects.requireNonNull(initParams);
		this.pointer = doInit(model, initParams);
		this.model = model;
		this.initParams = initParams;
		int poolingTypeCode = doGetPoolingType();
		poolingType = PoolingType.byCode(poolingTypeCode);
		contextSize = doGetContextSize();
		batchSize = doGetBatchSize();
	}

	/*
	 * NATIVE
	 */
	private static native long doInit(LlamaCppModel model, ContextParams params);

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

	public ContextParams getInitParams() {
		return initParams;
	}

	public PoolingType getPoolingType() {
		return poolingType;
	}

	public int getContextSize() {
		return contextSize;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public LlamaCppBatchProcessor getBatchProcessor() {
		return batchProcessor;
	}

	void setBatchProcessor(LlamaCppBatchProcessor batchProcessor) {
		if (batchProcessor != null)
			throw new IllegalArgumentException("A batch processor is already active for this context");
		this.batchProcessor = batchProcessor;
	}

	/*
	 * STATIC UTILTIES
	 */

	public static ContextParams defaultContextParams() {
		return DEFAULT_PARAMS;
	}

	/*
	 * CLASSES
	 */

}
