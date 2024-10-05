package org.argeo.jjml.llama;

import java.util.Objects;

public class LlamaCppContext {

	private LlamaCppModel model;

	private LlamaCppPoolingType poolingType;

	private LlamaCppContextParams initParams;

	// implementation
	private Long pointer;

	/*
	 * NATIVE
	 */
	native long doInit(LlamaCppModel model, LlamaCppContextParams params);

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
		pointer = doInit(model, params);
		this.initParams = params;
		int poolingTypeCode = doGetPoolingType();
		poolingType = LlamaCppPoolingType.byCode(poolingTypeCode);
	}

	public void destroy() {
		Objects.requireNonNull(pointer, "Context not initialized");
		doDestroy();
		pointer = null;
	}

	/*
	 * UTILITIES
	 */
	protected void checkNotInitialized() {
		if (pointer != null)
			throw new IllegalStateException("Context is already initialized, destroy it first.");
	}

	protected void checkInitialized() {
		if (pointer == null)
			throw new IllegalStateException("Context is not initialized.");
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

	long getPointer() {
		checkInitialized();
		return pointer;
	}

}
