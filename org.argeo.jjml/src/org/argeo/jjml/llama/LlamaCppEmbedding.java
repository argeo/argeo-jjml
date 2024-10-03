package org.argeo.jjml.llama;

import java.util.StringJoiner;

public class LlamaCppEmbedding {
	private final float[] vector;

	public LlamaCppEmbedding(float[] vector) {
		this.vector = vector;
	}

	public float[] getVector() {
		return vector;
	}

	@Override
	public String toString() {
		StringJoiner sj = new StringJoiner(", ", "[ ", " ...]");
		for (int i = 0; i < Math.min(10, vector.length); i++) {
			sj.add(Float.toString(vector[i]));
		}
		return sj.toString();
	}

}
