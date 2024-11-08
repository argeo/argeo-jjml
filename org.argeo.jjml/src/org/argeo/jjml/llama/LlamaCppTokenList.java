package org.argeo.jjml.llama;

import java.io.PrintStream;
import java.nio.IntBuffer;

@Deprecated
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

	IntBuffer asIntBuffer() {
		return IntBuffer.wrap(tokens);
	}

	public int size() {
		return tokens.length;
	}

	public void logIntegers(PrintStream out, String separator) {
		for (int i = 0; i < tokens.length; i++) {
			if (i != 0)
				out.append(separator);
			out.append(Integer.toString(tokens[i]));
		}
	}

	public String getAsText() {
		return model.getVocabulary().deTokenize(asIntBuffer(), true, true);
	}

	@Override
	public String toString() {
		return "TokenList model=" + model.getAsLong() + ", " + tokens.length + " tokens";
	}

}
