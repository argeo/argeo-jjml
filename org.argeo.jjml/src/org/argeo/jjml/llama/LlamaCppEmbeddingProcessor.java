package org.argeo.jjml.llama;

import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.List;

import org.argeo.jjml.llama.params.PoolingType;

/** Computes embeddings. */
public class LlamaCppEmbeddingProcessor {
	private final LlamaCppContext context;

	public LlamaCppEmbeddingProcessor(LlamaCppContext context) {
		this.context = context;
	}

	private static native void doProcessEmbeddings(long contextPointer, int[][] tokens, float[] emb);

	public float[][] processEmbeddings(List<String> prompts) {
		IntBuffer[] tokenLists = new IntBuffer[prompts.size()];
		for (int i = 0; i < prompts.size(); i++) {
			String prompt = prompts.get(i);
			IntBuffer tokenList = context.getModel().getVocabulary().tokenize(prompt);
			tokenLists[i] = tokenList;
		}
		return processEmbeddings(tokenLists);
	}

	public float[][] processEmbeddings(IntBuffer[] inputs) {
		// logic taken from llama.cpp's examples/embedding
		PoolingType poolingType = context.getPoolingType();
		int n_embd_count = 0;
		if (PoolingType.LLAMA_POOLING_TYPE_NONE.equals(poolingType)) {
			for (IntBuffer tokenList : inputs) {
				n_embd_count += tokenList.remaining();
			}
		} else {
			n_embd_count = inputs.length;
		}

		int n_embd = context.getModel().getEmbeddingSize();

		float[] emb = new float[n_embd_count * n_embd];
		Arrays.fill(emb, 0);

		int[][] tokens = new int[inputs.length][];
		for (int i = 0; i < inputs.length; i++) {
			// FIXME check buffer type
			tokens[i] = inputs[i].array();
		}
		doProcessEmbeddings(context.getAsLong(), tokens, emb);

		// TODO optimize storage, returned values, and copy
		float[][] res = new float[n_embd_count][];
		for (int j = 0;;) { // at least one iteration (one prompt)
			float[] arr = new float[n_embd];
			for (int i = 0;;) { // at least one iteration (n_embd > 0)
				arr[i] = emb[j * n_embd + i];
				i++;
				if (i == n_embd)
					break;
			}
			res[j] = arr;
			j++;
			if (j == n_embd_count)
				break;
		}
		return res;
	}

	/*
	 * ACCESSORS
	 */
	protected LlamaCppContext getContext() {
		return context;
	}
}
