package org.argeo.jjml.llama;

import java.util.Objects;

public class LlamaCppContext {

	private LlamaCppModel model;

	/*
	 * Parameters
	 */
	/** Text context size. Taken from model if 0 (default). */
	private int contextSize = 0;
	/** Logical maximum batch size (must be >=32 to use BLAS). */
	private int maximumBatchSize = 2048;

	// implementation
	private Long pointer;

	native long doInit(LlamaCppModel model);

	native void doDestroy();

	public void init() {
		Objects.requireNonNull(model, "Model must be set");
		pointer = doInit(model);
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

	public int getContextSize() {
		return contextSize;
	}

	public void setContextSize(int contextSize) {
		checkNotInitialized();
		this.contextSize = contextSize;
	}

	public int getMaximumBatchSize() {
		return maximumBatchSize;
	}

	public void setMaximumBatchSize(int maximumBatchSize) {
		checkNotInitialized();
		this.maximumBatchSize = maximumBatchSize;
	}

	long getPointer() {
		return pointer;
	}

}
