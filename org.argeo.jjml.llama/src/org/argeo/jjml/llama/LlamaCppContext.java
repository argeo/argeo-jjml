package org.argeo.jjml.llama;

import java.util.Objects;

public class LlamaCppContext {

	private LlamaCppModel model;

	private Long pointer;

	native long doInit(long modelPointer);

	native void doDestroy(long pointer);

	public void init() {
		Objects.requireNonNull(model, "Model must be set");
		pointer = doInit(model.getPointer());
	}

	public void destroy() {
		Objects.requireNonNull(pointer, "Context not initialized");
		doDestroy(pointer);
	}

	public LlamaCppModel getModel() {
		return model;
	}

	public void setModel(LlamaCppModel model) {
		this.model = model;
	}

}
