package org.argeo.jjml.llama;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LlamaCppEmbeddingProcessor {
	private LlamaCppContext context;

	private static native void doProcessEmbeddings(long contextPointer, int[][] tokens, float[] emb);

	public List<LlamaCppEmbedding> processEmbeddings(List<String> prompts) {
		List<LlamaCppTokenList> tokenLists = new ArrayList<>(prompts.size());
		for (String prompt : prompts) {
			LlamaCppTokenList tokenList = context.getModel().tokenize(prompt, true);
			tokenLists.add(tokenList);
		}

		// logic taken from llama.cpp's examples/embedding
		LlamaCppPoolingType poolingType = context.getPoolingType();
		int n_embd_count = 0;
		if (LlamaCppPoolingType.LLAMA_POOLING_TYPE_NONE.equals(poolingType)) {
			for (LlamaCppTokenList tokenList : tokenLists) {
				n_embd_count += tokenList.getTokens().length;
			}
		} else {
			n_embd_count = tokenLists.size();
		}

		int n_embd = context.getModel().getEmbeddingSize();

		float[] emb = new float[n_embd_count * n_embd];
		Arrays.fill(emb, 0);

		int[][] tokens = new int[tokenLists.size()][];
		for (int i = 0; i < tokenLists.size(); i++) {
			tokens[i] = tokenLists.get(i).getTokens();
		}
		doProcessEmbeddings(context.getAsLong(), tokens, emb);

		// TODO optimise storage, returned values, and copy
		List<LlamaCppEmbedding> res = new ArrayList<>(n_embd_count);
		for (int j = 0;;) { // at least one iteration (one prompt)
			float[] arr = new float[n_embd];
			for (int i = 0;;) { // at least one iteration (n_embd > 0)
				arr[i] = emb[j * n_embd + i];
				i++;
				if (i == n_embd)
					break;
			}
			res.add(new LlamaCppEmbedding(arr));
			j++;
			if (j == n_embd_count)
				break;
		}
		return res;
	}

	public void setContext(LlamaCppContext context) {
		this.context = context;
	}

}
