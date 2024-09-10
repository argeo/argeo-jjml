package org.argeo.jjml.llama;

public class LlamaCppTokenList {
	private LlamaCppModel model;

	private int[] tokens;

	@Override
	public String toString() {
		return model.doDeTokenize(tokens, false);
	}
}
