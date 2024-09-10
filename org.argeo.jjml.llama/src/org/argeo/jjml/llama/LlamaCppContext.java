package org.argeo.jjml.llama;

import java.util.Objects;

public class LlamaCppContext {

	private LlamaCppModel model;

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
	}

	public LlamaCppModel getModel() {
		return model;
	}

	public void setModel(LlamaCppModel model) {
		this.model = model;
	}

	long getPointer() {
		return pointer;
	}

}
