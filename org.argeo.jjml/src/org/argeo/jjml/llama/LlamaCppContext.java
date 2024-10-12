package org.argeo.jjml.llama;

import java.util.Objects;

/**
 * Access to a llama.cpp context
 * 
 * @see llama.h - llama_context
 */
public class LlamaCppContext extends NativeReference {

	private LlamaCppModel model;

	private LlamaCppPoolingType poolingType;

	private LlamaCppContextParams initParams;

	/*
	 * NATIVE
	 */
	native long doInit(LlamaCppModel model, LlamaCppContextParams params);

	@Override
	native void doDestroy();

	native int doGetPoolingType();

	/*
	 * LIFECYCLE
	 */
	public void init() {
		init(LlamaCppContextParams.defaultContextParams());
	}

	public void init(LlamaCppContextParams params) {
		Objects.requireNonNull(model, "Model must be set");
		long pointer = doInit(model, params);
		setPointer(pointer);
		this.initParams = params;
		int poolingTypeCode = doGetPoolingType();
		poolingType = LlamaCppPoolingType.byCode(poolingTypeCode);
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
}
