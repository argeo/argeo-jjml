package org.argeo.jjml.llama;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

public class LlamaCppBatchProcessor {
	private int parallel = 2;
	
	private LlamaCppModel model;

	public List<? extends CompletionStage<LlamaCppTokenList>> process(String systemPrompt,
			List<String> sequencePrompt, int predictSize) {
		List<CompletableFuture<LlamaCppTokenList>> res = new ArrayList<>(sequencePrompt.size());
		
		int maxKvSize = systemPrompt.length();
		for (int i = 0; i < sequencePrompt.size(); i++) {
			res.add(new CompletableFuture<>());
		}

		return res;
	}
}
