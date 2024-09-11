package org.argeo.jjml.llama;

public class LlamaCppTokenList {
	private final LlamaCppModel model;

	private final int[] tokens;

	public LlamaCppTokenList(LlamaCppModel model, int[] tokens) {
		this.model = model;
		this.tokens = tokens;
	}

	int[] getTokens() {
		return tokens;
	}
	
	public int size() {
		return tokens.length;
	}

	@Override
	public String toString() {
		return model.doDeTokenize(tokens, false);
	}

}
