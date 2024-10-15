package org.argeo.jjml.llama;

import java.util.Objects;

/**
 * Access to a llama.cpp context
 * 
 * @see llama.h - llama_context
 */
public class LlamaCppContext extends NativeReference {

	private LlamaCppModel model;

	private LlamaCppContextParams initParams;

	// Effective parameters
	private LlamaCppPoolingType poolingType;
	private int contextSize;

	/*
	 * NATIVE
	 */
	private native long doInit(long modelPointer, LlamaCppContextParams params);

	@Override
	native void doDestroy(long pointer);

	private native int doGetPoolingType(long pointer);

	private native int doGetContextSize(long pointer);

	/*
	 * LIFECYCLE
	 */
	public void init() {
		init(LlamaCppContextParams.defaultContextParams());
	}

	public void init(LlamaCppContextParams params) {
		Objects.requireNonNull(model, "Model must be set");
		long pointer = doInit(model.getPointer(), params);
		setPointer(pointer);
		this.initParams = params;
		int poolingTypeCode = doGetPoolingType(getPointer());
		poolingType = LlamaCppPoolingType.byCode(poolingTypeCode);
		contextSize = doGetContextSize(getPointer());
	}

	/*
	 * ACCESSORS
	 */

	public LlamaCppModel getModel() {
		return model;
	}

	public void setModel(LlamaCppModel model) {
		checkNotInitialized();
		this.model = model;
	}

	public LlamaCppContextParams getInitParams() {
		checkInitialized();
		return initParams;
	}

	public LlamaCppPoolingType getPoolingType() {
		checkInitialized();
		return poolingType;
	}

	public int getContextSize() {
		return contextSize;
	}

}
