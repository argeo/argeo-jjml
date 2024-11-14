package org.argeo.jjml.llama.util;

import java.nio.IntBuffer;
import java.util.Collections;
import java.util.function.Function;

import org.argeo.jjml.llama.LlamaCppContext;
import org.argeo.jjml.llama.LlamaCppEmbeddingProcessor;
import org.argeo.jjml.llama.LlamaCppVocabulary;
import org.argeo.jjml.llama.params.PoolingType;

/** Computes embeddings based on chunks of a given size. */
public class SimpleEmbedding extends LlamaCppEmbeddingProcessor implements Function<String, float[][]> {
	private final LlamaCppVocabulary vocabulary;

	private final int chunkSize;

	/**
	 * Constructor.
	 * 
	 * @param context   The context used to initialize this processor.
	 * @param chunkSize The size of the chunks. If <=0, the strings will be
	 *                  processed as a whole.
	 */
	public SimpleEmbedding(LlamaCppContext context, int chunkSize) {
		super(context);
		this.vocabulary = getContext().getModel().getVocabulary();
		this.chunkSize = chunkSize;
	}

	@Override
	public float[][] apply(String str) {
		if (chunkSize <= 0 || PoolingType.LLAMA_POOLING_TYPE_NONE.equals(getContext().getPoolingType())) {
			return processEmbeddings(Collections.singletonList(str));
		}
		int totalLength = str.length();
		IntBuffer[] inputs = new IntBuffer[totalLength / chunkSize + (totalLength % chunkSize == 0 ? 0 : 1)];
		for (int i = 0; i < inputs.length; i++) {
			String chunk;
			if (i == inputs.length - 1) {
				chunk = str.substring(i * chunkSize);
			} else {
				chunk = str.substring(i * chunkSize, (i + 1) * chunkSize);
			}
			inputs[i] = vocabulary.tokenize(chunk);
		}
		return processEmbeddings(inputs);
	}

}
